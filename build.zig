const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.option(std.builtin.OptimizeMode, "optimize", "Prioritize performance, safety, or binary size") orelse .ReleaseFast;

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
    configureCompileStep(b, exe, build_options, enable_metal);

    b.installArtifact(exe);

    const fixture_tool = b.addExecutable(.{
        .name = "make-tiny-fixture",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/make_tiny_fixture.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    configureCompileStep(b, fixture_tool, build_options, enable_metal);
    b.installArtifact(fixture_tool);

    const run_cmd = b.addRunArtifact(exe);
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run ziggy-llm");
    run_step.dependOn(&run_cmd.step);

    const fixture_run = b.addRunArtifact(fixture_tool);
    if (b.args) |args| {
        fixture_run.addArgs(args);
    }
    const fixture_step = b.step("tiny-fixture", "Write a reproducible ziggy-tiny GGUF fixture");
    fixture_step.dependOn(&fixture_run.step);

    const exe_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    configureCompileStep(b, exe_tests, build_options, enable_metal);

    const test_run = b.addRunArtifact(exe_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&test_run.step);

    const check_step = b.step("check", "Build the executable and run tests");
    check_step.dependOn(b.getInstallStep());
    check_step.dependOn(test_step);
}

fn configureCompileStep(
    b: *std.Build,
    compile: *std.Build.Step.Compile,
    build_options: *std.Build.Step.Options,
    enable_metal: bool,
) void {
    compile.root_module.addOptions("build_options", build_options);

    if (!enable_metal) return;
    const target = compile.root_module.resolved_target.?;
    if (target.result.os.tag != .macos or target.result.cpu.arch != .aarch64) return;

    compile.linkLibC();
    compile.root_module.addIncludePath(b.path("src/runtime/metal"));
    compile.root_module.addCSourceFile(.{
        .file = b.path("src/runtime/metal/bridge.m"),
        .flags = &.{"-fobjc-arc"},
    });
    compile.root_module.linkFramework("Foundation", .{});
    compile.root_module.linkFramework("Metal", .{});
}
