#include "interpreter/interpreter.h"
#include "interpreter/transforms.h"
#include "util/error_util.h"

#include <cmath>
#include <list>
#include <map>

namespace hannk {

Interpreter::Interpreter(std::unique_ptr<OpGroup> m, InterpreterOptions options)
    : model_(std::move(m)) {
    init(options);
}

Interpreter::~Interpreter() {
}

namespace {

struct TensorInfo {
    size_t offset = 0;
    size_t size = 0;
    int first_use = std::numeric_limits<int>::max();
    int last_use = std::numeric_limits<int>::min();

    // bool operator<(const TensorInfo &other) const {
    //     return offset < other.offset;
    // }
};

class FindAllocatableTensors : public OpVisitor {
    void process(TensorPtr t) {
        if (!t || t->is_external() || t->is_constant() || t->is_dynamic()) {
            return;
        }
        HCHECK(!t->is_allocated());
        auto &info = tensors[t.get()];
        HCHECK(info.size == 0 || info.size == t->buffer().size_in_bytes());
        info.size = t->buffer().size_in_bytes();
        info.first_use = std::min(info.first_use, op_index);
        info.last_use = std::max(info.last_use, op_index);
    }

    void visit(OpGroup *g) override {
        for (int i = 0; i < g->op_count(); i++) {
            op_index++;
            Op *op = g->op(i);
            for (int j = 0; j < op->input_count(); j++) {
                process(op->input(j));
            }
            for (int j = 0; j < op->output_count(); j++) {
                process(op->output(j));
            }
            op->accept(this);
        }
    }

public:
    std::map<Tensor *, TensorInfo> tensors;
    int op_index = -1;
};


class AllocateAll : public OpVisitor {
    void visit(OpGroup *g) {
        for (int i = 0; i < g->op_count(); i++) {
            Op *op = g->op(i);
            for (int j = 0; j < op->input_count(); j++) {
                op->input(j)->allocate();
            }
            for (int j = 0; j < op->output_count(); j++) {
                op->output(j)->allocate();
            }
            op->accept(this);
        }
    }
};

}  // namespace

void Interpreter::init(InterpreterOptions options) {
    pad_for_ops(model_.get());
    in_place(model_.get());
    fold_constants(model_.get());
    remove_dead_ops(model_.get());

    FindAllocatableTensors find_tensors;
    model_->accept(&find_tensors);
    std::cout << "Final op_count is " << find_tensors.op_index + 1 << "\n";
    std::cout << "Final allocatable tensor count is " << find_tensors.tensors.size() << "\n";
    for (const auto &it : find_tensors.tensors) {
        const auto t = it.first;
        const auto info = it.second;
        std::cout << "Tensor " << t->name() << " size " << info.size << " life [" << info.first_use << " ... " << info.last_use << "]\n";
    }

    // TODO: Find a better schedule for executing the ops, including
    // better lifetime management for these allocations.
    AllocateAll allocate_all;
    model_->accept(&allocate_all);
}

void Interpreter::execute() {
    model_->execute();
}

TensorPtr Interpreter::get_tensor(const std::string &name) {
    for (int i = 0; i < model_->op_count(); i++) {
        Op *op = model_->op(i);
        for (int j = 0; j < op->input_count(); j++) {
            if (op->input(j)->name() == name) {
                return op->input(j);
            }
        }
        for (int j = 0; j < op->output_count(); j++) {
            if (op->output(j)->name() == name) {
                return op->output(j);
            }
        }
    }
    return nullptr;
}

std::vector<TensorPtr> Interpreter::inputs() {
    std::vector<TensorPtr> result;
    for (int i = 0; i < model_->input_count(); i++) {
        result.push_back(model_->input(i));
    }

    return result;
}

std::vector<TensorPtr> Interpreter::outputs() {
    std::vector<TensorPtr> result;
    for (int i = 0; i < model_->output_count(); i++) {
        result.push_back(model_->output(i));
    }

    return result;
}

}  // namespace hannk
