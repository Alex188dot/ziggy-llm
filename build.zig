const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const default_enable_metal = target.result.os.tag == .macos and target.result.cpu.arch == .aarch64;
    const enable_metal = b.option(bool, "metal", "Enable the Metal backend") orelse default_enable_metal;

    const version = b.option([]const u8, "version", "Override the build version string") orelse "0.1.0-dev";

    const build_options = b.addOptions();
    build_options.addOption(bool, "enable_metal", enable_metal);
    build_options.addOption([]const u8, "version", version);

    const exe = b.addExecutable(.{
        .name = "ziggy-llm",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    exe.root_module.addOptions("build_options", build_options);

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run ziggy-llm");
    run_step.dependOn(&run_cmd.step);

    const exe_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    exe_tests.root_module.addOptions("build_options", build_options);

    const test_run = b.addRunArtifact(exe_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&test_run.step);

    const check_step = b.step("check", "Build the executable and run tests");
    check_step.dependOn(b.getInstallStep());
    check_step.dependOn(test_step);
}
