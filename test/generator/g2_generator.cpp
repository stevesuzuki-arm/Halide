#include "Halide.h"

#include "g2_generator.h"

namespace Halide {
namespace Testing {

// TODO: buffers. imageparams? outputbuffers?
// TODO: pass in targetinfo.
Var x, y;

Func g2_func_impl(Func input, Expr offset, int scaling) {
    Func output;
    output(x, y) = input(x, y) * scaling + offset;
    output.compute_root();

    return output;
}

Func g2_tuple_func_impl(Func input, Expr offset, int scaling) {
    Func output;
    output(x, y) = Tuple(input(x, y) * scaling + offset, cast<double>(input(x, y)) / scaling - offset);
    output.compute_root();

    return output;
}

const auto g2_lambda_impl = [](Func input, Expr offset, int scaling,
                               Type ignored_type, bool ignored_bool, std::string ignored_string, int8_t ignored_int8) {
    std::cout << "Ignoring type: " << ignored_type << "\n";
    std::cout << "Ignoring bool: " << (int)ignored_bool << "\n";
    std::cout << "Ignoring string: " << ignored_string << "\n";
    std::cout << "Ignoring int8: " << (int)ignored_int8 << "\n";

    Func output = g2_func_impl(input, offset, scaling);
    // TODO output.vectorize(x, Target::natural_vector_size<int32_t>());

    return output;
};

}  // namespace Testing
}  // namespace Halide

using namespace Halide;
using namespace Halide::Internal;

#define HALIDE_REGISTER_G2(GEN_FUNC, GEN_REGISTRY_NAME, GEN_BIND_INPUTS, GEN_BIND_OUTPUTS)                                                             \
    namespace halide_register_generator {                                                                                          \
    struct halide_global_ns;                                                                                                       \
    namespace GEN_REGISTRY_NAME##_ns {                                                                                             \
        std::unique_ptr<Halide::Internal::AbstractGenerator> factory(const Halide::GeneratorContext &context) {                    \
            using Input = Halide::Internal::FnBinder::Input;\
            using Output = Halide::Internal::FnBinder::Output;\
            using Constant = Halide::Internal::FnBinder::Constant;\
            Halide::Internal::FnBinder d(GEN_FUNC, GEN_BIND_INPUTS, GEN_BIND_OUTPUTS);                                                                                   \
            return G2GeneratorFactory(#GEN_REGISTRY_NAME, std::move(d))(context);                                                  \
        }                                                                                                                          \
    }                                                                                                                              \
    static auto reg_##GEN_REGISTRY_NAME = Halide::Internal::RegisterGenerator(#GEN_REGISTRY_NAME, GEN_FUNC);                       \
    }                                                                                                                              \
    static_assert(std::is_same<::halide_register_generator::halide_global_ns, halide_register_generator::halide_global_ns>::value, \
                  "HALIDE_REGISTER_G2 must be used at global scope");


#if 0
HALIDE_REGISTER_G2(g2_func_impl, g2, {
                FnBinder::Input("input", Int(32), 2),
                FnBinder::Input("offset", Int(32)),
                FnBinder::Constant("scaling", 2),
            },
            FnBinder::Output("output", Int(32), 2)
        )
#else
RegisterGenerator register_1(
    "g2",
    [](const GeneratorContext &context) -> std::unique_ptr<AbstractGenerator> {
        FnBinder d{
            Halide::Testing::g2_func_impl,
            {
                FnBinder::Input("input", Int(32), 2),
                FnBinder::Input("offset", Int(32)),
                FnBinder::Constant("scaling", 2),
            },
            FnBinder::Output("output", Int(32), 2),
        };
        return G2GeneratorFactory("g2", std::move(d))(context);
    });
#endif

RegisterGenerator register_2(
    "g2_lambda",
    [](const GeneratorContext &context) -> std::unique_ptr<AbstractGenerator> {
        FnBinder d{
            Halide::Testing::g2_lambda_impl,
            {
                FnBinder::Input("input", Int(32), 2),
                FnBinder::Input("offset", Int(32)),
                FnBinder::Constant("scaling", 2),
                FnBinder::Constant("ignored_type", Int(32)),
                FnBinder::Constant("ignored_bool", false),
                FnBinder::Constant("ignored_string", "qwerty"),
                FnBinder::Constant("ignored_int8", (int8_t)-27),
            },
            FnBinder::Output("output", Int(32), 2),
        };
        return G2GeneratorFactory("g2_lambda", std::move(d))(context);
    });

// RegisterGenerator register_3(
//     "g2_tuple",
//     [](const GeneratorContext &context) -> std::unique_ptr<AbstractGenerator> {
//         FnBinder d{
//             Halide::Testing::g2_tuple_func_impl,
//             {
//                 FnBinder::Input("input", Int(32), 2),
//                 FnBinder::Input("offset", Int(32)),
//                 FnBinder::Constant("scaling", 2),
//                 FnBinder::Constant("ignored_type", Int(32)),
//                 FnBinder::Constant("ignored_bool", false),
//                 FnBinder::Constant("ignored_string", "qwerty"),
//                 FnBinder::Constant("ignored_int8", (int8_t)-27),
//             },
//             FnBinder::Output("output", {Int(32), Float(64)}, 2),
//         };
//         return G2GeneratorFactory("g2_tuple", std::move(d))(context);
//     });
