const std = @import("std");
const mod = @import("mod.zig");
const gguf = @import("../../gguf.zig");

pub const RegistryError = error{
    FamilyNotRegistered,
    UnsupportedFamily,
};

const MaxFamilies = 16;

pub const FamilyRegistry = struct {
    families: [MaxFamilies]?mod.FamilyHandler = .{null} ** MaxFamilies,
    count: usize = 0,

    pub fn register(self: *FamilyRegistry, handler: mod.FamilyHandler) RegistryError!void {
        if (self.count >= MaxFamilies) {
            return RegistryError.UnsupportedFamily;
        }
        for (self.families) |existing| {
            if (existing) |fam| {
                if (std.meta.eql(fam.family, handler.family)) {
                    return RegistryError.FamilyNotRegistered;
                }
            }
        }
        self.families[self.count] = handler;
        self.count += 1;
    }

    pub fn getRuntime(self: *const FamilyRegistry, family: mod.ModelFamily) ?mod.FamilyRuntime {
        for (self.families) |handler| {
            if (handler) |fam| {
                if (std.meta.eql(fam.family, family)) {
                    return fam.runtime;
                }
            }
        }
        return null;
    }

    pub fn getHandler(self: *const FamilyRegistry, family: mod.ModelFamily) ?mod.FamilyHandler {
        for (self.families) |handler| {
            if (handler) |fam| {
                if (std.meta.eql(fam.family, family)) {
                    return fam;
                }
            }
        }
        return null;
    }

    pub fn detectAndGetRuntime(self: *const FamilyRegistry, gguf_report: gguf.InspectReport) ?mod.FamilyRuntime {
        const family = mod.detectModelFamily(gguf_report.architecture);
        return self.getRuntime(family);
    }
};

pub fn createDefaultRegistry() FamilyRegistry {
    return FamilyRegistry{};
}

test "registry starts empty" {
    var registry = FamilyRegistry{};
    try std.testing.expectEqual(@as(usize, 0), registry.count);
    try std.testing.expect(registry.getRuntime(.llama) == null);
}

test "registry can register a family" {
    var registry = FamilyRegistry{};
    const dummy_runtime = mod.FamilyRuntime{
        .ctx = null,
        .generate_fn = struct {
            fn f(ctx: ?*anyopaque, allocator: std.mem.Allocator, model_path: []const u8, prompt: []const u8, options: mod.FamilyGenerateOptions) mod.FamilyError!mod.FamilyReport {
                _ = ctx;
                _ = allocator;
                _ = model_path;
                _ = prompt;
                _ = options;
                unreachable;
            }
        }.f,
        .deinit_fn = struct {
            fn f(ctx: ?*anyopaque) void {
                _ = ctx;
            }
        }.f,
    };
    const handler = mod.FamilyHandler{
        .family = .llama,
        .capabilities = .{
            .supports_cpu = true,
            .supports_metal = true,
            .supported_quant_types = &.{},
            .max_context_length = 8192,
        },
        .runtime = dummy_runtime,
    };
    try registry.register(handler);
    try std.testing.expectEqual(@as(usize, 1), registry.count);
    try std.testing.expect(registry.getRuntime(.llama) != null);
}

test "registry returns null for unregistered family" {
    var registry = FamilyRegistry{};
    try std.testing.expect(registry.getRuntime(.qwen) == null);
}
