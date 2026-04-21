const std = @import("std");
const cli = @import("cli.zig");
const runtime = @import("runtime.zig");
const gguf = @import("gguf.zig");
const moon_quant = @import("moon_quant.zig");
const chat_runtime = @import("chat_runtime.zig");
const server = @import("server.zig");
const server_runtime = @import("server_runtime.zig");
const build_options = @import("build_options");
const update = @import("update.zig");

pub fn dispatch(writer: *std.Io.Writer, allocator: std.mem.Allocator, config: cli.Config) !void {
    switch (config.command) {
        .help => try cli.printHelp(writer),
        .version => try writer.print("ziggy-llm {s}\n", .{build_options.version}),
        .run => try runModel(writer, allocator, config),
        .chat => try chat_runtime.runChat(writer, allocator, config),
        .bench => try benchModel(writer, allocator, config),
        .inspect => try printInspect(writer, config),
        .serve => try server_runtime.serve(writer, allocator, config),
        .update => try update.runUpdateCommand(writer, allocator),
    }
}

fn runModel(writer: *std.Io.Writer, allocator: std.mem.Allocator, config: cli.Config) !void {
    const model_path = config.model_path orelse return error.MissingModelPath;
    const prompt = config.prompt orelse return error.MissingPrompt;

    try runtime.runCommand(writer, allocator, model_path, prompt, .{
        .max_tokens = config.max_tokens,
        .context_length = config.context_length,
        .seed = config.seed,
        .temperature = config.temperature,
        .repeat_penalty = config.repeat_penalty,
        .top_k = config.top_k,
        .top_p = config.top_p,
        .min_p = config.min_p,
        .backend = config.backend,
        .moon_quant = config.moon_quant,
        .metal_profile = config.metal_profile,
        .sampling_strategy = config.sampling_strategy,
    });
}

fn benchModel(writer: *std.Io.Writer, allocator: std.mem.Allocator, config: cli.Config) !void {
    const model_path = config.model_path orelse return error.MissingModelPath;
    const prompt = config.prompt orelse return error.MissingPrompt;

    try runtime.benchCommand(writer, allocator, model_path, prompt, .{
        .max_tokens = config.max_tokens,
        .context_length = config.context_length,
        .seed = config.seed,
        .temperature = config.temperature,
        .repeat_penalty = config.repeat_penalty,
        .top_k = config.top_k,
        .top_p = config.top_p,
        .min_p = config.min_p,
        .backend = config.backend,
        .moon_quant = config.moon_quant,
        .metal_profile = config.metal_profile,
        .sampling_strategy = config.sampling_strategy,
    }, config.bench_runs);
}

fn printInspect(writer: *std.Io.Writer, config: cli.Config) !void {
    const model_path = config.model_path orelse return error.MissingModelPath;

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    const report = try gguf.inspectFile(arena.allocator(), model_path);
    try gguf.printInspectReport(writer, model_path, report);
    try moon_quant.printInspectSummary(writer, report);

    if (config.dump_tensors) {
        try writer.print("\n", .{});
        try inspectDetailed(writer, arena.allocator(), model_path, true);
    }
}

fn inspectDetailed(writer: *std.Io.Writer, allocator: std.mem.Allocator, model_path: []const u8, dump_tensors: bool) !void {
    const file = try std.fs.cwd().openFile(model_path, .{});
    defer file.close();

    const stat = try file.stat();
    var reader = ParsingReader{ .allocator = allocator, .file = file, .pos = 0 };

    const magic = try readExactString(&reader, 4);
    if (!std.mem.eql(u8, &magic, "GGUF")) return error.InvalidMagic;
    try writer.print("Magic: {s}\n", .{magic});

    const version = try reader.readInt(u32);
    try writer.print("Version: {d}\n", .{version});

    const tensor_count = try reader.readInt(u64);
    try writer.print("Tensor count: {d}\n", .{tensor_count});

    const metadata_count = try reader.readInt(u64);
    try writer.print("Metadata count: {d}\n\n", .{metadata_count});

    try writer.print("=== METADATA ===\n", .{});
    var metadata_idx: u64 = 0;
    while (metadata_idx < metadata_count) : (metadata_idx += 1) {
        const key = try readLengthPrefixedString(allocator, &reader);
        const value_type = try reader.readInt(u32);
        try writer.print("  {s}: type={d}", .{ key, value_type });

        switch (@as(gguf.ValueType, @enumFromInt(value_type))) {
            .uint8 => {
                const v = try reader.readInt(u8);
                try writer.print(" value={d}\n", .{v});
            },
            .int8 => {
                const v = try reader.readInt(u8);
                try writer.print(" value={d}\n", .{v});
            },
            .uint16 => {
                const v = try reader.readInt(u16);
                try writer.print(" value={d}\n", .{v});
            },
            .int16 => {
                const v = try reader.readInt(u16);
                try writer.print(" value={d}\n", .{v});
            },
            .uint32 => {
                const v = try reader.readInt(u32);
                try writer.print(" value={d}\n", .{v});
            },
            .int32 => {
                const v = try reader.readInt(u32);
                try writer.print(" value={d}\n", .{v});
            },
            .float32 => {
                const raw = try reader.readInt(u32);
                const v = @as(f32, @bitCast(raw));
                try writer.print(" value={d}\n", .{v});
            },
            .bool => {
                const raw = try reader.readInt(u8);
                try writer.print(" value={s}\n", .{if (raw == 1) "true" else "false"});
            },
            .string => {
                const len = try reader.readInt(u64);
                const bytes = try reader.readBytes(len);
                try writer.print(" value=\"{s}\"\n", .{bytes});
            },
            .uint64 => {
                const v = try reader.readInt(u64);
                try writer.print(" value={d}\n", .{v});
            },
            .int64 => {
                const v = try reader.readInt(u64);
                try writer.print(" value={d}\n", .{v});
            },
            .float64 => {
                const raw = try reader.readInt(u64);
                const v = @as(f64, @bitCast(raw));
                try writer.print(" value={d}\n", .{v});
            },
            .array => {
                const element_type = try reader.readInt(u32);
                const count = try reader.readInt(u64);
                try writer.print(" [array of type={d}, count={d}]\n", .{ element_type, count });
                var elem_idx: u64 = 0;
                while (elem_idx < count) : (elem_idx += 1) {
                    const elem_raw_type = @as(gguf.ValueType, @enumFromInt(element_type));
                    switch (elem_raw_type) {
                        .string => {
                            const len = try reader.readInt(u64);
                            const bytes = try reader.readBytes(len);
                            defer allocator.free(bytes);
                            try writer.print("    [{d}] \"{s}\"\n", .{ elem_idx, bytes });
                        },
                        .uint8, .int8 => {
                            const v = try reader.readInt(u8);
                            try writer.print("    [{d}] {d}\n", .{ elem_idx, v });
                        },
                        .uint16, .int16 => {
                            const v = try reader.readInt(u16);
                            try writer.print("    [{d}] {d}\n", .{ elem_idx, v });
                        },
                        .uint32, .int32 => {
                            const v = try reader.readInt(u32);
                            try writer.print("    [{d}] {d}\n", .{ elem_idx, v });
                        },
                        .float32 => {
                            const raw = try reader.readInt(u32);
                            const v = @as(f32, @bitCast(raw));
                            try writer.print("    [{d}] {d}\n", .{ elem_idx, v });
                        },
                        .uint64, .int64 => {
                            const raw = try reader.readInt(u64);
                            const v = @as(f64, @bitCast(raw));
                            try writer.print("    [{d}] {d}\n", .{ elem_idx, v });
                        },
                        .float64 => {
                            const raw = try reader.readInt(u64);
                            const v = @as(f64, @bitCast(raw));
                            try writer.print("    [{d}] {d}\n", .{ elem_idx, v });
                        },
                        .bool => {
                            const raw = try reader.readInt(u8);
                            try writer.print("    [{d}] {s}\n", .{ elem_idx, if (raw == 1) "true" else "false" });
                        },
                        else => {
                            try writer.print("    [{d}] <unhandled array element type={d}>\n", .{ elem_idx, element_type });
                        },
                    }
                }
            },
        }
        allocator.free(key);
    }

    if (dump_tensors) {
        try writer.print("\n=== TENSORS ===\n", .{});
        var tensor_idx: u64 = 0;
        while (tensor_idx < tensor_count) : (tensor_idx += 1) {
            const name = try readLengthPrefixedString(allocator, &reader);
            const n_dims = try reader.readInt(u32);
            var dims: [4]u64 = undefined;
            var d: usize = 0;
            while (d < n_dims) : (d += 1) {
                dims[d] = try reader.readInt(u64);
            }
            const tensor_type = try reader.readInt(u32);
            const offset = try reader.readInt(u64);

            try writer.print("  [{d}] {s} dims={d} [", .{ tensor_idx, name, n_dims });
            var dim_i: usize = 0;
            while (dim_i < n_dims) : (dim_i += 1) {
                if (dim_i > 0) try writer.print(", ", .{});
                try writer.print("{d}", .{dims[dim_i]});
            }
            try writer.print("] type={d} offset={d}\n", .{ tensor_type, offset });
            allocator.free(name);
        }
    }

    try writer.print("\n=== FILE STATS ===\n", .{});
    try writer.print("File size: {d} bytes\n", .{stat.size});
}

const ParsingReader = struct {
    allocator: std.mem.Allocator,
    file: std.fs.File,
    pos: u64 = 0,

    fn readInt(self: *ParsingReader, comptime T: type) !T {
        var buffer: [@sizeOf(T)]u8 = undefined;
        const actual = try self.file.preadAll(&buffer, self.pos);
        if (actual != buffer.len) return error.TruncatedFile;
        self.pos += buffer.len;
        return std.mem.readInt(T, &buffer, .little);
    }

    fn readBytes(self: *ParsingReader, len: usize) ![]u8 {
        const pos = self.pos;
        const bytes = try self.allocator.alloc(u8, len);
        errdefer self.allocator.free(bytes);
        const actual = try self.file.preadAll(bytes, pos);
        if (actual != len) return error.TruncatedFile;
        self.pos = pos + len;
        return bytes;
    }
};

fn readExactString(reader: *ParsingReader, comptime len: usize) ![len]u8 {
    var buffer: [len]u8 = undefined;
    const actual = try reader.file.preadAll(&buffer, reader.pos);
    if (actual != len) return error.TruncatedFile;
    reader.pos += len;
    return buffer;
}

fn readLengthPrefixedString(allocator: std.mem.Allocator, reader: *ParsingReader) ![]u8 {
    const len = try reader.readInt(u64);
    const bytes = try reader.readBytes(len);
    defer allocator.free(bytes);
    const owned = try allocator.dupe(u8, bytes);
    return owned;
}
