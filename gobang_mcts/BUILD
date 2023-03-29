package(default_visibility = ["//visibility:public"])

cc_library(
    name = "gobang_env",
    hdrs = ["gobang_env.hpp"],
)

cc_test(
    name = "gobang_env_test",
    srcs = ["gobang_env_test.cc"],
    deps = [
        ":gobang_env",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "mcts",
    srcs = ["mcts.hpp"],
    deps = [
        ":gobang_env",
    ],
)

cc_test(
    name = "mcts_test",
    srcs = ["mcts_test.cc"],
    deps = [
        ":mcts",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "gobang_mcts",
    srcs = ["gobang_mcts.hpp"],
    deps = [
        ":gobang_env",
        ":mcts",
    ],
)

cc_test(
    name = "gobang_mcts_test",
    srcs = ["gobang_mcts_test.cc"],
    deps = [
        ":gobang_mcts",
        "@com_google_googletest//:gtest_main",
    ],
)
