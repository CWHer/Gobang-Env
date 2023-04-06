load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "utils",
    hdrs = ["utils.hpp"],
)

cc_library(
    name = "gobang_env",
    hdrs = ["gobang_env.hpp"],
    deps = [
        ":utils",
    ],
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
    hdrs = ["mcts.hpp"],
    deps = [
        ":utils",
    ],
)

cc_test(
    name = "mcts_test",
    srcs = ["mcts_test.cc"],
    deps = [
        ":gobang_env",
        ":mcts",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "gobang_selfplay",
    hdrs = ["gobang_selfplay.hpp"],
    deps = [
        ":gobang_env",
        ":mcts",
        ":utils",
    ],
)

cc_test(
    name = "gobang_selfplay_test",
    srcs = ["gobang_selfplay_test.cc"],
    deps = [
        ":gobang_selfplay",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "gobang_envpool",
    hdrs = ["gobang_envpool.hpp"],
    deps = [
        ":gobang_selfplay",
        ":utils",
        "//envpool/core:async_envpool",
    ],
)

cc_test(
    name = "gobang_envpool_test",
    srcs = ["gobang_envpool_test.cc"],
    deps = [
        ":gobang_envpool",
        "@com_google_googletest//:gtest_main",
    ],
)

pybind_extension(
    name = "py_gobang_envpool",
    srcs = [
        "gobang_envpool.cc",
    ],
    deps = [
        ":gobang_envpool",
        "//envpool/core:py_envpool",
    ],
)

py_library(
    name = "py_gobang_envpool_init",
    srcs = ["__init__.py"],
    data = [":py_gobang_envpool.so"],
    deps = [
        "//envpool/python:api",
    ],
)

py_library(
    name = "py_gobang_envpool_registration",
    srcs = ["registration.py"],
    deps = [
        "//envpool:registration",
    ],
)
