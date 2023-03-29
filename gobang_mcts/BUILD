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

