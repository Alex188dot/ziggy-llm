// Template Family Module - mod.zig
//
// Copy this file to src/runtime/families/<your_family>/mod.zig
// and update to export your FamilyHandler.
//
// Example for adding a new family "phi":
// -------------------------------------------------------------------
// pub const runtime = @import("runtime.zig");
// pub const FamilyHandler = runtime.createPhiFamilyHandler();
// -------------------------------------------------------------------
//
// Then in src/runtime/mod.zig:
//   const phi_family = @import("families/phi/mod.zig");
//   reg.register(phi_family.FamilyHandler) catch unreachable;

pub const runtime = @import("runtime.zig");
pub const FamilyHandler = runtime.createTemplateFamilyHandler();
