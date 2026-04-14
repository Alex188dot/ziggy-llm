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
    initialized: bool = false,

    pub fn init(self: *FamilyRegistry) void {
        if (self.initialized) return;
        self.initialized = true;
    }

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

var global_registry: FamilyRegistry = FamilyRegistry{};
var global_registry_initialized = false;

pub fn getGlobalRegistry() *FamilyRegistry {
    if (!global_registry_initialized) {
        global_registry.init();
        global_registry_initialized = true;
    }
    return &global_registry;
}

pub fn createDefaultRegistry() FamilyRegistry {
    return FamilyRegistry{};
}
