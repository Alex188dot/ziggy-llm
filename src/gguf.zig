const std = @import("std");

pub const supported_versions = "GGUF v2 and v3 (little-endian)";
pub const supported_metadata_fields =
    "general.type=model, general.architecture, general.alignment, general.file_type, general.quantization_version, tokenizer.ggml.model, tokenizer.ggml.pre, tokenizer.ggml.tokens, tokenizer.ggml.{bos,eos,unk,pad}_token_id, tokenizer.ggml.add_{bos,eos}_token, tokenizer.chat_template";

const gguf_magic = "GGUF";
const default_alignment: u32 = 32;
const max_tensor_dims: usize = 4;
const ggml_type_count: usize = 41;

const Parser = struct {
    file: std.fs.File,
    pos: u64 = 0,

    fn readBytes(self: *Parser, buffer: []u8) !void {
        const actual = try self.file.preadAll(buffer, self.pos);
        if (actual != buffer.len) return error.TruncatedFile;
        self.pos = try std.math.add(u64, self.pos, buffer.len);
    }

    fn readInt(self: *Parser, comptime T: type) !T {
        var buffer: [@sizeOf(T)]u8 = undefined;
        try self.readBytes(&buffer);
        return std.mem.readInt(T, &buffer, .little);
    }

    fn skipBytes(self: *Parser, bytes: u64) !void {
        self.pos = try std.math.add(u64, self.pos, bytes);
    }
};

const ValueType = enum(u32) {
    uint8 = 0,
    int8 = 1,
    uint16 = 2,
    int16 = 3,
    uint32 = 4,
    int32 = 5,
    float32 = 6,
    bool = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    int64 = 11,
    float64 = 12,
};

pub const TensorType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
    iq2_xxs = 16,
    iq2_xs = 17,
    iq3_xxs = 18,
    iq1_s = 19,
    iq4_nl = 20,
    iq3_s = 21,
    iq2_s = 22,
    iq4_xs = 23,
    i8 = 24,
    i16 = 25,
    i32 = 26,
    i64 = 27,
    f64 = 28,
    iq1_m = 29,
    bf16 = 30,
    tq1_0 = 34,
    tq2_0 = 35,
    mxfp4 = 39,
    nvfp4 = 40,
};

pub const ChatTemplateStyle = enum {
    generic,
    chatml,
    qwen,
    tinyllama,
    llama3,
};

/// Model family classification for routing different inference paths
pub const ModelFamily = enum {
    /// Unknown or generic model family
    unknown,
    /// Qwen models (Qwen1.5, Qwen2, etc.) - use Qwen-specific Metal buffer allocation
    qwen,
    /// TinyLlama 1.0 - uses <|system|>/<|user|>/<|assistant|> with </s> end markers
    tinyllama,
    /// Llama 3.0+ (including Llama 3.1, 3.2) - uses <|begin_of_text|>/<|start_header_id|>/<|eot_id|>
    llama3,
    /// Other Llama models (Llama 2, early Llama 3 betas) - generic chat format
    llama,
    /// Gemma models from Google
    gemma,
    /// Mistral models
    mistral,
    /// Phi models from Microsoft
    phi,
    /// Mixtral mixture of experts
    mixtral,
};

const TypeLayout = struct {
    block_size: u16,
    type_size: u16,
};

pub const InspectReport = struct {
    version: u32,
    metadata_count: u64,
    tensor_count: u64,
    alignment: u32,
    artifact_type: []const u8,
    architecture: []const u8,
    quantization_version: ?u32,
    file_type: ?u32,
    dominant_tensor_type: TensorType,
    dominant_tensor_count: u64,
    data_offset: u64,
    tokenizer_model: ?[]const u8,
    tokenizer_pre: ?[]const u8,
    tokenizer_tokens: ?u64,
    chat_template: ?[]const u8,
    bos_token_id: ?u32,
    eos_token_id: ?u32,
    unk_token_id: ?u32,
    pad_token_id: ?u32,
    add_bos_token: ?bool,
    add_eos_token: ?bool,

    pub fn fileTypeName(self: InspectReport) []const u8 {
        if (self.file_type) |value| return formatFileType(value);
        return "unknown";
    }

    pub fn dominantTensorTypeName(self: InspectReport) []const u8 {
        return formatTensorType(self.dominant_tensor_type);
    }

    /// Detect chat template style using both template content and architecture fallback
    pub fn chatTemplateStyle(self: InspectReport) ChatTemplateStyle {
        // First priority: template-based detection
        const template_style = detectChatTemplateStyle(self.chat_template);
        if (template_style != .generic) return template_style;

        // Second priority: architecture-based fallback
        return detectStyleByArchitecture(self.architecture);
    }

    /// Classify model by its architecture name
    pub fn modelFamily(self: InspectReport) ModelFamily {
        return classifyModelFamily(self.architecture, self.chat_template);
    }

    /// Check if this model requires Qwen-specific Metal buffer allocation
    pub fn requiresQwenMetalPath(self: InspectReport) bool {
        return self.modelFamily() == .qwen;
    }

    /// Check if this model uses Apple Metal GPU backend
    pub fn usesMetalGpu(self: InspectReport) bool {
        return self.modelFamily() != .qwen;
    }
};

/// Architecture-based template style detection (fallback when template detection fails)
fn detectStyleByArchitecture(architecture: []const u8) ChatTemplateStyle {
    // Llama 3.x uses different template format
    if (std.mem.startsWith(u8, architecture, "llama")) {
        // Llama 3+ uses <|begin_of_text|> marker
        return .llama3;
    }

    // Qwen models
    if (std.mem.startsWith(u8, architecture, "qwen")) {
        return .qwen;
    }

    // TinyLlama is a separate architecture
    if (std.mem.startsWith(u8, architecture, "tinyllama")) {
        return .tinyllama;
    }

    return .generic;
}

/// Classify model into its family based on architecture and template
fn classifyModelFamily(architecture: []const u8, chat_template: ?[]const u8) ModelFamily {
    const template = chat_template orelse "";

    // Qwen family
    if (std.mem.startsWith(u8, architecture, "qwen")) {
        return .qwen;
    }

    // TinyLlama family (distinct from general llama)
    if (std.mem.startsWith(u8, architecture, "tinyllama")) {
        return .tinyllama;
    }

    // Llama 3.x family (uses <|begin_of_text|>/<|eot_id|> markers)
    if (std.mem.startsWith(u8, architecture, "llama")) {
        if (std.mem.indexOf(u8, template, "<|begin_of_text|>") != null or
            std.mem.indexOf(u8, template, "<|eot_id|>") != null)
        {
            return .llama3;
        }
        return .llama;
    }

    // Gemma family
    if (std.mem.startsWith(u8, architecture, "gemma")) {
        return .gemma;
    }

    // Mistral family
    if (std.mem.startsWith(u8, architecture, "mistral")) {
        return .mistral;
    }

    // Phi family
    if (std.mem.startsWith(u8, architecture, "phi")) {
        return .phi;
    }

    // Mixtral family
    if (std.mem.startsWith(u8, architecture, "mixtral")) {
        return .mixtral;
    }

    return .unknown;
}

pub const InspectError = error{
    InvalidMagic,
    UnsupportedVersion,
    UnsupportedEndianness,
    UnsupportedArtifactType,
    TruncatedFile,
    InvalidMetadataType,
    InvalidMetadataValue,
    MissingRequiredMetadata,
    InvalidTensorMetadata,
    Overflow,
};

pub fn inspectFile(allocator: std.mem.Allocator, model_path: []const u8) !InspectReport {
    const file = try std.fs.cwd().openFile(model_path, .{});
    defer file.close();

    const stat = try file.stat();
    var reader = Parser{ .file = file };

    const magic = readExact(&reader, 4) catch return error.TruncatedFile;
    if (!std.mem.eql(u8, &magic, gguf_magic)) {
        if (std.mem.eql(u8, &magic, "FUGG")) return error.UnsupportedEndianness;
        return error.InvalidMagic;
    }

    const version = readInt(&reader, u32) catch return error.TruncatedFile;
    if (version != 2 and version != 3) return error.UnsupportedVersion;

    const tensor_count = readInt(&reader, u64) catch return error.TruncatedFile;
    const metadata_count = readInt(&reader, u64) catch return error.TruncatedFile;

    var state = ParseState{};
    var kv_index: u64 = 0;
    while (kv_index < metadata_count) : (kv_index += 1) {
        try parseMetadataEntry(allocator, &reader, &state);
    }

    if (state.artifact_type) |artifact_type| {
        if (!std.mem.eql(u8, artifact_type, "model")) return error.UnsupportedArtifactType;
    }

    const architecture = state.architecture orelse return error.MissingRequiredMetadata;
    const artifact_type = state.artifact_type orelse "model";
    const alignment = state.alignment orelse default_alignment;
    if (alignment == 0 or !std.math.isPowerOfTwo(alignment)) return error.InvalidMetadataValue;

    var dominant_tensor_type: ?TensorType = null;
    var dominant_tensor_count: u64 = 0;

    var tensor_index: u64 = 0;
    while (tensor_index < tensor_count) : (tensor_index += 1) {
        const tensor = try parseTensorInfo(allocator, &reader, alignment);
        const tensor_bytes = try tensorByteSize(&tensor);
        const type_index: usize = @intCast(@intFromEnum(tensor.tensor_type));
        state.tensor_type_counts[type_index] += 1;

        if (state.tensor_type_counts[type_index] > dominant_tensor_count or dominant_tensor_type == null) {
            dominant_tensor_count = state.tensor_type_counts[type_index];
            dominant_tensor_type = tensor.tensor_type;
        }

        const tensor_end = try std.math.add(u64, tensor.offset, tensor_bytes);
        if (tensor_end > state.max_tensor_extent) state.max_tensor_extent = tensor_end;
    }

    if (tensor_count > 0 and dominant_tensor_type == null) return error.InvalidTensorMetadata;

    const after_metadata = reader.pos;
    const data_offset = alignForwardU64(after_metadata, alignment);
    if (data_offset > stat.size) return error.TruncatedFile;

    const data_bytes = stat.size - data_offset;
    if (state.max_tensor_extent > data_bytes) return error.TruncatedFile;

    return .{
        .version = version,
        .metadata_count = metadata_count,
        .tensor_count = tensor_count,
        .alignment = alignment,
        .artifact_type = artifact_type,
        .architecture = architecture,
        .quantization_version = state.quantization_version,
        .file_type = state.file_type,
        .dominant_tensor_type = dominant_tensor_type orelse .f32,
        .dominant_tensor_count = dominant_tensor_count,
        .data_offset = data_offset,
        .tokenizer_model = state.tokenizer_model,
        .tokenizer_pre = state.tokenizer_pre,
        .tokenizer_tokens = state.tokenizer_tokens,
        .chat_template = state.chat_template,
        .bos_token_id = state.bos_token_id,
        .eos_token_id = state.eos_token_id,
        .unk_token_id = state.unk_token_id,
        .pad_token_id = state.pad_token_id,
        .add_bos_token = state.add_bos_token,
        .add_eos_token = state.add_eos_token,
    };
}

pub fn printInspectReport(writer: *std.io.Writer, model_path: []const u8, report: InspectReport) !void {
    var quantization_version_buffer: [32]u8 = undefined;
    var tokenizer_tokens_buffer: [32]u8 = undefined;
    var bos_buffer: [32]u8 = undefined;
    var eos_buffer: [32]u8 = undefined;
    var unk_buffer: [32]u8 = undefined;
    var pad_buffer: [32]u8 = undefined;

    try writer.print(
        \\model: {s}
        \\gguf_version: {d}
        \\artifact_type: {s}
        \\architecture: {s}
        \\model_family: {s}
        \\tensor_count: {d}
        \\metadata_count: {d}
        \\alignment: {d}
        \\quantization: {s}
        \\dominant_tensor_type: {s} ({d} tensors)
        \\quantization_version: {s}
        \\tokenizer_model: {s}
        \\tokenizer_pre: {s}
        \\tokenizer_tokens: {s}
        \\tokenizer_special_tokens: bos={s} eos={s} unk={s} pad={s}
        \\tokenizer_add_bos: {s}
        \\tokenizer_add_eos: {s}
        \\chat_template_style: {s}
        \\data_offset: {d}
        \\
    ,
        .{
            model_path,
            report.version,
            report.artifact_type,
            report.architecture,
            @tagName(report.modelFamily()),
            report.tensor_count,
            report.metadata_count,
            report.alignment,
            report.fileTypeName(),
            report.dominantTensorTypeName(),
            report.dominant_tensor_count,
            optionalInt(&quantization_version_buffer, report.quantization_version),
            optionalString(report.tokenizer_model),
            optionalString(report.tokenizer_pre),
            optionalInt(&tokenizer_tokens_buffer, report.tokenizer_tokens),
            optionalInt(&bos_buffer, report.bos_token_id),
            optionalInt(&eos_buffer, report.eos_token_id),
            optionalInt(&unk_buffer, report.unk_token_id),
            optionalInt(&pad_buffer, report.pad_token_id),
            formatOptionalBool(report.add_bos_token),
            formatOptionalBool(report.add_eos_token),
            @tagName(report.chatTemplateStyle()),
            report.data_offset,
        },
    );
}

const ParseState = struct {
    artifact_type: ?[]const u8 = null,
    architecture: ?[]const u8 = null,
    alignment: ?u32 = null,
    quantization_version: ?u32 = null,
    file_type: ?u32 = null,
    tokenizer_model: ?[]const u8 = null,
    tokenizer_pre: ?[]const u8 = null,
    tokenizer_tokens: ?u64 = null,
    chat_template: ?[]const u8 = null,
    bos_token_id: ?u32 = null,
    eos_token_id: ?u32 = null,
    unk_token_id: ?u32 = null,
    pad_token_id: ?u32 = null,
    add_bos_token: ?bool = null,
    add_eos_token: ?bool = null,
    tensor_type_counts: [ggml_type_count]u64 = [_]u64{0} ** ggml_type_count,
    max_tensor_extent: u64 = 0,
};

const TensorInfo = struct {
    name: []const u8,
    dimensions: [max_tensor_dims]u64,
    n_dimensions: usize,
    tensor_type: TensorType,
    offset: u64,
};

fn parseMetadataEntry(
    allocator: std.mem.Allocator,
    reader: anytype,
    state: *ParseState,
) !void {
    const key = try readOwnedString(allocator, reader);
    const raw_value_type = readInt(reader, u32) catch return error.TruncatedFile;
    const value_type = std.meta.intToEnum(ValueType, raw_value_type) catch return error.InvalidMetadataType;

    if (std.mem.eql(u8, key, "general.type")) {
        state.artifact_type = try readExpectedString(allocator, reader, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "general.architecture")) {
        state.architecture = try readExpectedString(allocator, reader, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "general.alignment")) {
        state.alignment = try readExpectedUnsigned(u32, reader, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "general.quantization_version")) {
        state.quantization_version = try readExpectedUnsigned(u32, reader, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "general.file_type")) {
        state.file_type = try readExpectedUnsigned(u32, reader, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "tokenizer.ggml.model")) {
        state.tokenizer_model = try readExpectedString(allocator, reader, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "tokenizer.ggml.pre")) {
        state.tokenizer_pre = try readExpectedString(allocator, reader, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "tokenizer.ggml.tokens")) {
        state.tokenizer_tokens = try readStringArrayCount(reader, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "tokenizer.chat_template")) {
        state.chat_template = try readExpectedString(allocator, reader, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "tokenizer.ggml.bos_token_id")) {
        state.bos_token_id = try readExpectedUnsigned(u32, reader, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "tokenizer.ggml.eos_token_id")) {
        state.eos_token_id = try readExpectedUnsigned(u32, reader, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "tokenizer.ggml.unknown_token_id")) {
        state.unk_token_id = try readExpectedUnsigned(u32, reader, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "tokenizer.ggml.padding_token_id")) {
        state.pad_token_id = try readExpectedUnsigned(u32, reader, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "tokenizer.ggml.add_bos_token")) {
        state.add_bos_token = try readExpectedBool(reader, value_type);
        return;
    }
    if (std.mem.eql(u8, key, "tokenizer.ggml.add_eos_token")) {
        state.add_eos_token = try readExpectedBool(reader, value_type);
        return;
    }

    try skipValue(reader, value_type);
}

fn parseTensorInfo(
    allocator: std.mem.Allocator,
    reader: anytype,
    alignment: u32,
) !TensorInfo {
    const name = try readOwnedString(allocator, reader);
    if (name.len == 0) return error.InvalidTensorMetadata;

    const n_dimensions_u32 = readInt(reader, u32) catch return error.TruncatedFile;
    if (n_dimensions_u32 == 0 or n_dimensions_u32 > max_tensor_dims) return error.InvalidTensorMetadata;
    const n_dimensions: usize = @intCast(n_dimensions_u32);

    var dimensions = [_]u64{1} ** max_tensor_dims;
    for (0..n_dimensions) |index| {
        dimensions[index] = readInt(reader, u64) catch return error.TruncatedFile;
        if (dimensions[index] == 0) return error.InvalidTensorMetadata;
    }

    const raw_tensor_type = readInt(reader, u32) catch return error.TruncatedFile;
    const tensor_type = std.meta.intToEnum(TensorType, raw_tensor_type) catch return error.InvalidTensorMetadata;
    const offset = readInt(reader, u64) catch return error.TruncatedFile;

    if (offset % alignment != 0) return error.InvalidTensorMetadata;

    return .{
        .name = name,
        .dimensions = dimensions,
        .n_dimensions = n_dimensions,
        .tensor_type = tensor_type,
        .offset = offset,
    };
}

fn readInt(reader: anytype, comptime T: type) !T {
    return reader.readInt(T);
}

fn readExact(reader: anytype, comptime len: usize) ![len]u8 {
    var bytes: [len]u8 = undefined;
    try reader.readBytes(&bytes);
    return bytes;
}

fn readOwnedString(allocator: std.mem.Allocator, reader: anytype) ![]const u8 {
    const length_u64 = readInt(reader, u64) catch return error.TruncatedFile;
    const length: usize = std.math.cast(usize, length_u64) orelse return error.Overflow;
    const bytes = try allocator.alloc(u8, length);
    try reader.readBytes(bytes);
    return bytes;
}

fn readExpectedString(allocator: std.mem.Allocator, reader: anytype, value_type: ValueType) ![]const u8 {
    if (value_type != .string) return error.InvalidMetadataType;
    return readOwnedString(allocator, reader);
}

fn readExpectedUnsigned(comptime T: type, reader: anytype, value_type: ValueType) !T {
    return switch (value_type) {
        .uint8 => try castUnsigned(T, readInt(reader, u8) catch return error.TruncatedFile),
        .uint16 => try castUnsigned(T, readInt(reader, u16) catch return error.TruncatedFile),
        .uint32 => try castUnsigned(T, readInt(reader, u32) catch return error.TruncatedFile),
        .uint64 => try castUnsigned(T, readInt(reader, u64) catch return error.TruncatedFile),
        else => error.InvalidMetadataType,
    };
}

fn readExpectedBool(reader: anytype, value_type: ValueType) !bool {
    if (value_type != .bool) return error.InvalidMetadataType;
    const raw = readInt(reader, u8) catch return error.TruncatedFile;
    return switch (raw) {
        0 => false,
        1 => true,
        else => error.InvalidMetadataValue,
    };
}

fn readStringArrayCount(reader: anytype, value_type: ValueType) !u64 {
    if (value_type != .array) return error.InvalidMetadataType;
    const raw_element_type = readInt(reader, u32) catch return error.TruncatedFile;
    const element_type = std.meta.intToEnum(ValueType, raw_element_type) catch return error.InvalidMetadataType;
    if (element_type != .string) return error.InvalidMetadataType;
    const length = readInt(reader, u64) catch return error.TruncatedFile;

    var index: u64 = 0;
    while (index < length) : (index += 1) {
        const string_len = readInt(reader, u64) catch return error.TruncatedFile;
        try reader.skipBytes(string_len);
    }

    return length;
}

fn skipValue(reader: anytype, value_type: ValueType) !void {
    switch (value_type) {
        .uint8, .int8, .bool => try reader.skipBytes(1),
        .uint16, .int16 => try reader.skipBytes(2),
        .uint32, .int32, .float32 => try reader.skipBytes(4),
        .uint64, .int64, .float64 => try reader.skipBytes(8),
        .string => {
            const len = readInt(reader, u64) catch return error.TruncatedFile;
            try reader.skipBytes(len);
        },
        .array => {
            const raw_element_type = readInt(reader, u32) catch return error.TruncatedFile;
            const element_type = std.meta.intToEnum(ValueType, raw_element_type) catch return error.InvalidMetadataType;
            if (element_type == .array) return error.InvalidMetadataType;
            const length = readInt(reader, u64) catch return error.TruncatedFile;

            var index: u64 = 0;
            while (index < length) : (index += 1) {
                try skipValue(reader, element_type);
            }
        },
    }
}

fn castUnsigned(comptime T: type, value: anytype) !T {
    return std.math.cast(T, value) orelse error.InvalidMetadataValue;
}

fn tensorByteSize(tensor: *const TensorInfo) !u64 {
    const layout = typeLayout(tensor.tensor_type) orelse return error.InvalidTensorMetadata;
    const ne0 = tensor.dimensions[0];
    if (ne0 == 0) return error.InvalidTensorMetadata;
    if (ne0 % layout.block_size != 0) return error.InvalidTensorMetadata;

    const blocks = ne0 / layout.block_size;
    const row_size = try std.math.mul(u64, blocks, layout.type_size);

    var total_rows: u64 = 1;
    var index: usize = 1;
    while (index < tensor.n_dimensions) : (index += 1) {
        total_rows = try std.math.mul(u64, total_rows, tensor.dimensions[index]);
    }

    return try std.math.mul(u64, row_size, total_rows);
}

fn typeLayout(tensor_type: TensorType) ?TypeLayout {
    return switch (tensor_type) {
        .f32 => .{ .block_size = 1, .type_size = 4 },
        .f16 => .{ .block_size = 1, .type_size = 2 },
        .q4_0 => .{ .block_size = 32, .type_size = 18 },
        .q4_1 => .{ .block_size = 32, .type_size = 20 },
        .q5_0 => .{ .block_size = 32, .type_size = 22 },
        .q5_1 => .{ .block_size = 32, .type_size = 24 },
        .q8_0 => .{ .block_size = 32, .type_size = 34 },
        .q8_1 => .{ .block_size = 32, .type_size = 40 },
        .q2_k => .{ .block_size = 256, .type_size = 84 },
        .q3_k => .{ .block_size = 256, .type_size = 110 },
        .q4_k => .{ .block_size = 256, .type_size = 144 },
        .q5_k => .{ .block_size = 256, .type_size = 176 },
        .q6_k => .{ .block_size = 256, .type_size = 210 },
        .q8_k => .{ .block_size = 256, .type_size = 292 },
        .iq2_xxs => .{ .block_size = 256, .type_size = 66 },
        .iq2_xs => .{ .block_size = 256, .type_size = 74 },
        .iq3_xxs => .{ .block_size = 256, .type_size = 98 },
        .iq1_s => .{ .block_size = 256, .type_size = 50 },
        .iq4_nl => .{ .block_size = 32, .type_size = 18 },
        .iq3_s => .{ .block_size = 256, .type_size = 102 },
        .iq2_s => .{ .block_size = 256, .type_size = 82 },
        .iq4_xs => .{ .block_size = 256, .type_size = 136 },
        .i8 => .{ .block_size = 1, .type_size = 1 },
        .i16 => .{ .block_size = 1, .type_size = 2 },
        .i32 => .{ .block_size = 1, .type_size = 4 },
        .i64 => .{ .block_size = 1, .type_size = 8 },
        .f64 => .{ .block_size = 1, .type_size = 8 },
        .iq1_m => .{ .block_size = 256, .type_size = 56 },
        .bf16 => .{ .block_size = 1, .type_size = 2 },
        .tq1_0 => .{ .block_size = 256, .type_size = 54 },
        .tq2_0 => .{ .block_size = 256, .type_size = 66 },
        .mxfp4 => .{ .block_size = 32, .type_size = 17 },
        .nvfp4 => .{ .block_size = 64, .type_size = 36 },
    };
}

fn alignForwardU64(value: u64, alignment: u32) u64 {
    const a = @as(u64, alignment);
    return (value + a - 1) & ~(a - 1);
}

fn formatTensorType(tensor_type: TensorType) []const u8 {
    return switch (tensor_type) {
        .f32 => "F32",
        .f16 => "F16",
        .q4_0 => "Q4_0",
        .q4_1 => "Q4_1",
        .q5_0 => "Q5_0",
        .q5_1 => "Q5_1",
        .q8_0 => "Q8_0",
        .q8_1 => "Q8_1",
        .q2_k => "Q2_K",
        .q3_k => "Q3_K",
        .q4_k => "Q4_K",
        .q5_k => "Q5_K",
        .q6_k => "Q6_K",
        .q8_k => "Q8_K",
        .iq2_xxs => "IQ2_XXS",
        .iq2_xs => "IQ2_XS",
        .iq3_xxs => "IQ3_XXS",
        .iq1_s => "IQ1_S",
        .iq4_nl => "IQ4_NL",
        .iq3_s => "IQ3_S",
        .iq2_s => "IQ2_S",
        .iq4_xs => "IQ4_XS",
        .i8 => "I8",
        .i16 => "I16",
        .i32 => "I32",
        .i64 => "I64",
        .f64 => "F64",
        .iq1_m => "IQ1_M",
        .bf16 => "BF16",
        .tq1_0 => "TQ1_0",
        .tq2_0 => "TQ2_0",
        .mxfp4 => "MXFP4",
        .nvfp4 => "NVFP4",
    };
}

fn formatFileType(file_type: u32) []const u8 {
    return switch (file_type) {
        0 => "ALL_F32",
        1 => "MOSTLY_F16",
        2 => "MOSTLY_Q4_0",
        3 => "MOSTLY_Q4_1",
        7 => "MOSTLY_Q8_0",
        8 => "MOSTLY_Q5_0",
        9 => "MOSTLY_Q5_1",
        10 => "MOSTLY_Q2_K",
        11 => "MOSTLY_Q3_K_S",
        12 => "MOSTLY_Q3_K_M",
        13 => "MOSTLY_Q3_K_L",
        14 => "MOSTLY_Q4_K_S",
        15 => "MOSTLY_Q4_K_M",
        16 => "MOSTLY_Q5_K_S",
        17 => "MOSTLY_Q5_K_M",
        18 => "MOSTLY_Q6_K",
        19 => "MOSTLY_IQ2_XXS",
        20 => "MOSTLY_IQ2_XS",
        21 => "MOSTLY_Q2_K_S",
        22 => "MOSTLY_IQ3_XS",
        23 => "MOSTLY_IQ3_XXS",
        24 => "MOSTLY_IQ1_S",
        25 => "MOSTLY_IQ4_NL",
        26 => "MOSTLY_IQ3_S",
        27 => "MOSTLY_IQ3_M",
        28 => "MOSTLY_IQ2_S",
        29 => "MOSTLY_IQ2_M",
        30 => "MOSTLY_IQ4_XS",
        31 => "MOSTLY_IQ1_M",
        32 => "MOSTLY_BF16",
        36 => "MOSTLY_TQ1_0",
        37 => "MOSTLY_TQ2_0",
        38 => "MOSTLY_MXFP4_MOE",
        39 => "MOSTLY_NVFP4",
        else => "unknown",
    };
}

fn optionalString(value: ?[]const u8) []const u8 {
    return value orelse "<unset>";
}

/// Template-based chat template style detection
/// Priority: Llama3 > Qwen > TinyLlama > ChatML > Generic
pub fn detectChatTemplateStyle(chat_template: ?[]const u8) ChatTemplateStyle {
    const template = chat_template orelse return .generic;

    // Llama 3.x: uses <|begin_of_text|> and <|eot_id|> markers
    if (std.mem.indexOf(u8, template, "<|begin_of_text|>") != null) {
        return .llama3;
    }

    // Qwen: uses <|im_start|>/<|im_end|> markers (with im_ prefix)
    if (std.mem.indexOf(u8, template, "<|im_start|>") != null) {
        return .qwen;
    }

    // TinyLlama: uses <|system|>/<|user|>/<|assistant|> WITHOUT im_ prefix
    // Check for <|system|> specifically (TinyLlama v1.0 template)
    if (std.mem.indexOf(u8, template, "<|system|>") != null) {
        return .tinyllama;
    }

    // ChatML: uses <|user|>/<|assistant|> without <|system|> (or with different format)
    // But only if they DON'T have the im_ prefix (which would be Qwen)
    if (std.mem.indexOf(u8, template, "<|user|>") != null and
        std.mem.indexOf(u8, template, "<|assistant|>") != null)
    {
        // Double-check it's not Qwen format
        if (std.mem.indexOf(u8, template, "<|im_") == null) {
            return .chatml;
        }
    }

    return .generic;
}

fn formatOptionalBool(value: ?bool) []const u8 {
    if (value) |v| return if (v) "true" else "false";
    return "<unset>";
}

fn optionalInt(buffer: []u8, value: anytype) []const u8 {
    if (value) |v| return std.fmt.bufPrint(buffer, "{d}", .{v}) catch "<format-error>";
    return "<unset>";
}

test "inspect parses a valid GGUF v3 fixture" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const fixture = try makeFixture(.{});
    defer std.testing.allocator.free(fixture);
    try writeFixtureFile(tmp.dir, "valid-v3.gguf", fixture);
    const path = try tmp.dir.realpathAlloc(allocator, "valid-v3.gguf");

    const report = try inspectFile(allocator, path);
    try std.testing.expectEqual(@as(u32, 3), report.version);
    try std.testing.expectEqualStrings("model", report.artifact_type);
    try std.testing.expectEqualStrings("llama", report.architecture);
    try std.testing.expectEqual(@as(u64, 1), report.tensor_count);
    try std.testing.expectEqual(@as(u64, 3), report.tokenizer_tokens.?);
    try std.testing.expectEqual(@as(u32, 1), report.file_type.?);
    try std.testing.expectEqual(TensorType.f16, report.dominant_tensor_type);
}

test "inspect parses a valid GGUF v2 fixture" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const fixture = try makeFixture(.{
        .version = 2,
        .file_type = 12,
        .tensor_type = .q4_k,
        .dimensions = .{ 256, 1, 1, 1 },
    });
    defer std.testing.allocator.free(fixture);
    try writeFixtureFile(tmp.dir, "valid-v2.gguf", fixture);
    const path = try tmp.dir.realpathAlloc(allocator, "valid-v2.gguf");

    const report = try inspectFile(allocator, path);
    try std.testing.expectEqual(@as(u32, 2), report.version);
    try std.testing.expectEqual(@as(u32, 12), report.file_type.?);
    try std.testing.expectEqual(TensorType.q4_k, report.dominant_tensor_type);
}

test "inspect rejects unsupported artifact types" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const fixture = try makeFixture(.{
        .general_type = "adapter",
    });
    defer std.testing.allocator.free(fixture);
    try writeFixtureFile(tmp.dir, "adapter.gguf", fixture);
    const path = try tmp.dir.realpathAlloc(allocator, "adapter.gguf");

    try std.testing.expectError(error.UnsupportedArtifactType, inspectFile(allocator, path));
}

test "inspect rejects missing required architecture metadata" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const fixture = try makeFixture(.{
        .include_architecture = false,
    });
    defer std.testing.allocator.free(fixture);
    try writeFixtureFile(tmp.dir, "missing-arch.gguf", fixture);
    const path = try tmp.dir.realpathAlloc(allocator, "missing-arch.gguf");

    try std.testing.expectError(error.MissingRequiredMetadata, inspectFile(allocator, path));
}

test "inspect rejects malformed tensor metadata" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const fixture = try makeFixture(.{
        .dimensions = .{ 0, 3, 1, 1 },
    });
    defer std.testing.allocator.free(fixture);
    try writeFixtureFile(tmp.dir, "bad-tensor.gguf", fixture);
    const path = try tmp.dir.realpathAlloc(allocator, "bad-tensor.gguf");

    try std.testing.expectError(error.InvalidTensorMetadata, inspectFile(allocator, path));
}

test "inspect rejects truncated GGUF data" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const fixture = try makeFixture(.{
        .truncate_tensor_data = true,
    });
    defer std.testing.allocator.free(fixture);
    try writeFixtureFile(tmp.dir, "truncated.gguf", fixture);
    const path = try tmp.dir.realpathAlloc(allocator, "truncated.gguf");

    try std.testing.expectError(error.TruncatedFile, inspectFile(allocator, path));
}

test "inspect rejects invalid magic" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeFixtureFile(tmp.dir, "bad-magic.gguf", "NOPE");

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const path = try tmp.dir.realpathAlloc(allocator, "bad-magic.gguf");

    try std.testing.expectError(error.InvalidMagic, inspectFile(allocator, path));
}

test "inspect parses chat template metadata and detects chatml" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const fixture = try makeFixture(.{
        .chat_template =
        \\{% for message in messages %}
        \\{{ '<|user|>\n' + message['content'] + eos_token }}
        \\{% endfor %}
        \\{{ '<|assistant|>' }}
        ,
    });
    defer std.testing.allocator.free(fixture);
    try writeFixtureFile(tmp.dir, "chat-template.gguf", fixture);
    const path = try tmp.dir.realpathAlloc(allocator, "chat-template.gguf");

    const report = try inspectFile(allocator, path);
    try std.testing.expect(report.chat_template != null);
    try std.testing.expectEqual(ChatTemplateStyle.chatml, report.chatTemplateStyle());
}

test "detectChatTemplateStyle identifies llama3 templates" {
    const template = "{{ '<|begin_of_text|>' }}{% for message in messages %}{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}";
    try std.testing.expectEqual(ChatTemplateStyle.llama3, detectChatTemplateStyle(template));
}

test "detectChatTemplateStyle identifies qwen templates" {
    const template = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}{% endfor %}{{ '<|im_start|>assistant\n' }}";
    try std.testing.expectEqual(ChatTemplateStyle.qwen, detectChatTemplateStyle(template));
}

test "detectChatTemplateStyle identifies tinyllama templates" {
    const template = "<|system|>\n{system}</s>\n{% for message in messages %}<|{{ message['role'] }}|>\n{{ message['content'] }}</s>\n{% endfor %}<|assistant|>\n";
    try std.testing.expectEqual(ChatTemplateStyle.tinyllama, detectChatTemplateStyle(template));
}

test "detectChatTemplateStyle falls back to architecture for llama" {
    // No template - should fall back to llama3 for llama architecture
    try std.testing.expectEqual(ChatTemplateStyle.llama3, detectChatTemplateByArchitectureFallback("llama", null));
}

test "modelFamily classifies qwen correctly" {
    try std.testing.expectEqual(ModelFamily.qwen, classifyModelFamily("qwen2", null));
    try std.testing.expectEqual(ModelFamily.qwen, classifyModelFamily("qwen1.5", null));
}

test "modelFamily classifies tinyllama correctly" {
    try std.testing.expectEqual(ModelFamily.tinyllama, classifyModelFamily("tinyllama", null));
}

test "modelFamily classifies llama3 correctly" {
    const llama3_template = "<|begin_of_text|>...";
    try std.testing.expectEqual(ModelFamily.llama3, classifyModelFamily("llama", llama3_template));
}

test "modelFamily classifies gemma correctly" {
    try std.testing.expectEqual(ModelFamily.gemma, classifyModelFamily("gemma", null));
    try std.testing.expectEqual(ModelFamily.gemma, classifyModelFamily("gemma2", null));
}

// Helper for testing architecture fallback
fn detectChatTemplateByArchitectureFallback(architecture: []const u8, chat_template: ?[]const u8) ChatTemplateStyle {
    const template_style = detectChatTemplateStyle(chat_template);
    if (template_style != .generic) return template_style;
    return detectStyleByArchitecture(architecture);
}

const FixtureOptions = struct {
    version: u32 = 3,
    general_type: []const u8 = "model",
    include_architecture: bool = true,
    architecture: []const u8 = "llama",
    alignment: u32 = default_alignment,
    file_type: u32 = 1,
    quantization_version: u32 = 2,
    tokenizer_model: []const u8 = "gpt2",
    tokenizer_pre: []const u8 = "default",
    chat_template: ?[]const u8 = null,
    tensor_type: TensorType = .f16,
    dimensions: [4]u64 = .{ 32, 3, 1, 1 },
    truncate_tensor_data: bool = false,
};

fn makeFixture(options: FixtureOptions) ![]const u8 {
    var list: std.ArrayList(u8) = .empty;
    defer list.deinit(std.testing.allocator);
    try list.ensureTotalCapacity(std.testing.allocator, 2048);

    writeBytes(&list, gguf_magic);
    writeInt(&list, u32, options.version);

    const kv_count: u64 = (if (options.include_architecture) @as(u64, 11) else 10) +
        (if (options.chat_template != null) @as(u64, 1) else 0);
    writeInt(&list, u64, 1);
    writeInt(&list, u64, kv_count);

    writeStringKv(&list, "general.type", options.general_type);
    if (options.include_architecture) writeStringKv(&list, "general.architecture", options.architecture);
    writeU32Kv(&list, "general.alignment", options.alignment);
    writeU32Kv(&list, "general.file_type", options.file_type);
    writeU32Kv(&list, "general.quantization_version", options.quantization_version);
    writeStringKv(&list, "tokenizer.ggml.model", options.tokenizer_model);
    writeStringKv(&list, "tokenizer.ggml.pre", options.tokenizer_pre);
    writeStringArrayKv(&list, "tokenizer.ggml.tokens", &.{ "<unk>", "<s>", "</s>" });
    if (options.chat_template) |chat_template| writeStringKv(&list, "tokenizer.chat_template", chat_template);
    writeU32Kv(&list, "tokenizer.ggml.bos_token_id", 1);
    writeU32Kv(&list, "tokenizer.ggml.eos_token_id", 2);
    writeBoolKv(&list, "tokenizer.ggml.add_bos_token", true);

    writeString(&list, "token_embd.weight");
    writeInt(&list, u32, 2);
    writeInt(&list, u64, options.dimensions[0]);
    writeInt(&list, u64, options.dimensions[1]);
    writeInt(&list, u32, @intFromEnum(options.tensor_type));
    writeInt(&list, u64, 0);

    const aligned_metadata_size = alignForwardU64(list.items.len, options.alignment);
    const tensor_bytes = if (options.dimensions[0] == 0)
        @as(u64, 0)
    else
        tensorByteSize(&.{
            .name = "token_embd.weight",
            .dimensions = options.dimensions,
            .n_dimensions = 2,
            .tensor_type = options.tensor_type,
            .offset = 0,
        }) catch 0;

    const padding_len = aligned_metadata_size - list.items.len;
    for (0..padding_len) |_| list.appendAssumeCapacity(0);

    const bytes_to_write = if (options.truncate_tensor_data and tensor_bytes > 0) tensor_bytes - 1 else tensor_bytes;
    for (0..bytes_to_write) |_| list.appendAssumeCapacity(0);

    return list.toOwnedSlice(std.testing.allocator);
}

fn writeFixtureFile(dir: std.fs.Dir, name: []const u8, contents: []const u8) !void {
    const file = try dir.createFile(name, .{});
    defer file.close();
    try file.writeAll(contents);
}

fn writeBytes(list: *std.ArrayList(u8), bytes: []const u8) void {
    list.appendSliceAssumeCapacity(bytes);
}

fn writeInt(list: *std.ArrayList(u8), comptime T: type, value: T) void {
    var bytes: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, &bytes, value, .little);
    list.appendSliceAssumeCapacity(&bytes);
}

fn writeString(list: *std.ArrayList(u8), value: []const u8) void {
    writeInt(list, u64, value.len);
    list.appendSliceAssumeCapacity(value);
}

fn writeStringKv(list: *std.ArrayList(u8), key: []const u8, value: []const u8) void {
    writeString(list, key);
    writeInt(list, u32, @intFromEnum(ValueType.string));
    writeString(list, value);
}

fn writeU32Kv(list: *std.ArrayList(u8), key: []const u8, value: u32) void {
    writeString(list, key);
    writeInt(list, u32, @intFromEnum(ValueType.uint32));
    writeInt(list, u32, value);
}

fn writeBoolKv(list: *std.ArrayList(u8), key: []const u8, value: bool) void {
    writeString(list, key);
    writeInt(list, u32, @intFromEnum(ValueType.bool));
    writeInt(list, u8, if (value) 1 else 0);
}

fn writeStringArrayKv(list: *std.ArrayList(u8), key: []const u8, values: []const []const u8) void {
    writeString(list, key);
    writeInt(list, u32, @intFromEnum(ValueType.array));
    writeInt(list, u32, @intFromEnum(ValueType.string));
    writeInt(list, u64, values.len);
    for (values) |value| writeString(list, value);
}
