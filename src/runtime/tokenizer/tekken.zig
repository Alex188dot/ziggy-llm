const std = @import("std");
const json = std.json;

pub const TekkenConfig = struct {
    pattern: []const u8,
    num_vocab_tokens: u32,
    default_vocab_size: u32,
    default_num_special_tokens: u32,
    version: []const u8,
};

pub const TekkenVocabEntry = struct {
    rank: u32,
    token_bytes: []const u8,
    token_str: []const u8,
};

pub const TekkenTokenizer = struct {
    config: TekkenConfig,
    vocab: []TekkenVocabEntry,
    token_to_id: std.StringHashMap(u32),
    bos_token_id: ?u32 = null,
    eos_token_id: ?u32 = null,
    pad_token_id: ?u32 = null,
    unk_token_id: ?u32 = null,

    pub fn deinit(self: *TekkenTokenizer, allocator: std.mem.Allocator) void {
        allocator.free(self.config.pattern);
        allocator.free(self.config.version);
        for (self.vocab) |entry| {
            allocator.free(entry.token_bytes);
            allocator.free(entry.token_str);
        }
        allocator.free(self.vocab);
        self.token_to_id.deinit();
        self.* = undefined;
    }

    pub fn encode(self: *const TekkenTokenizer, allocator: std.mem.Allocator, text: []const u8, out: []u32) !usize {
        _ = allocator;
        _ = self;
        _ = text;
        _ = out;
        return error.NotYetImplemented;
    }

    pub fn decode(self: *const TekkenTokenizer, allocator: std.mem.Allocator, token_ids: []const u32, out: []u8) !usize {
        _ = allocator;
        _ = self;
        _ = token_ids;
        _ = out;
        return error.NotYetImplemented;
    }
};

pub fn loadTekkenTokenizer(allocator: std.mem.Allocator, tekken_path: []const u8) !TekkenTokenizer {
    const file = try std.fs.cwd().openFile(tekken_path, .{});
    defer file.close();

    const file_size = try file.getEndPos();
    const contents = try file.readAllAlloc(allocator, file_size);
    defer allocator.free(contents);

    const parsed = try json.parseFromSlice(json.Value, allocator, contents, .{});
    defer parsed.deinit();

    const root = parsed.value;
    const config_obj = root.object.get("config") orelse return error.InvalidFormat;
    const config_value = config_obj.value;
    const vocab_arr = root.object.get("vocab") orelse return error.InvalidFormat;
    const vocab_value = vocab_arr.value;

    const pattern = config_value.object.get("pattern") orelse return error.InvalidFormat;
    const num_vocab = config_value.object.get("num_vocab_tokens") orelse return error.InvalidFormat;
    const default_vocab_size = config_value.object.get("default_vocab_size") orelse return error.InvalidFormat;
    const default_num_special = config_value.object.get("default_num_special_tokens") orelse return error.StandardJsonPathMemberNotFound;
    const version = config_value.object.get("version") orelse return error.InvalidFormat;

    const config = TekkenConfig{
        .pattern = try allocator.dupe(u8, pattern.string),
        .num_vocab_tokens = @intCast(num_vocab.integer),
        .default_vocab_size = @intCast(default_vocab_size.integer),
        .default_num_special_tokens = @intCast(default_num_special.integer),
        .version = try allocator.dupe(u8, version.string),
    };

    const vocab_array = vocab_value.array;
    const vocab = try allocator.alloc(TekkenVocabEntry, vocab_array.items.len);
    errdefer allocator.free(vocab);

    var token_to_id = std.StringHashMap(u32).init(allocator);
    errdefer token_to_id.deinit();

    for (vocab_array.items, 0..) |entry_value, i| {
        const entry_obj = entry_value.object;
        const rank_val = entry_obj.get("rank") orelse return error.InvalidFormat;
        const token_bytes_val = entry_obj.get("token_bytes") orelse return error.InvalidFormat;
        const token_str_val = entry_obj.get("token_str") orelse return error.InvalidFormat;

        const rank: u32 = @intCast(rank_val.integer);
        const token_bytes = try allocator.dupe(u8, token_bytes_val.string);
        const token_str = try allocator.dupe(u8, token_str_val.string);

        vocab[i] = .{
            .rank = rank,
            .token_bytes = token_bytes,
            .token_str = token_str,
        };

        try token_to_id.put(token_str, rank);
    }

    return .{
        .config = config,
        .vocab = vocab,
        .token_to_id = token_to_id,
    };
}

pub fn tekkenBosTokenId(self: *TekkenTokenizer) ?u32 {
    return self.bos_token_id;
}

pub fn tekkenEosTokenId(self: *TekkenTokenizer) ?u32 {
    return self.eos_token_id;
}
