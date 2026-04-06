const std = @import("std");
const build_options = @import("build_options");

pub fn checkForUpdates(allocator: std.mem.Allocator) void {
    const thread = std.Thread.spawn(.{}, checkBackground, .{allocator}) catch return;
    thread.detach();
}

fn checkBackground(allocator: std.mem.Allocator) void {
    const latest_version = fetchLatestVersion(allocator) catch return;
    defer allocator.free(latest_version);

    var current_version = build_options.version;
    if (std.mem.startsWith(u8, current_version, "v")) current_version = current_version[1..];

    var latest_clean = latest_version;
    if (std.mem.startsWith(u8, latest_clean, "v")) latest_clean = latest_clean[1..];

    const current_sv = std.SemanticVersion.parse(current_version) catch return;
    const latest_sv = std.SemanticVersion.parse(latest_clean) catch return;

    if (latest_sv.order(current_sv) == .gt) {
        var stderr_buffer: [512]u8 = undefined;
        var stderr = std.fs.File.stderr().writer(&stderr_buffer);
        stderr.interface.print("\n\x1b[95mYou are using an older version of ziggy-llm (v{s}), update to v{s} now, by running: ziggy-llm update\x1b[0m\n\n", .{ current_version, latest_clean }) catch {};
    }
}

fn fetchLatestVersion(allocator: std.mem.Allocator) ![]u8 {
    var client: std.http.Client = .{ .allocator = allocator };
    defer client.deinit();

    var transfer_buffer: [8 * 1024]u8 = undefined;
    var reader_buffer: [8 * 1024]u8 = undefined;

    const uri = try std.Uri.parse("https://api.github.com/repos/Alex188dot/ziggy-llm/releases/latest");
    var request = try client.request(.GET, uri, .{
        .headers = .{
            .user_agent = .{ .override = "ziggy-llm" },
            .accept_encoding = .{ .override = "application/vnd.github.v3+json" },
        },
    });
    defer request.deinit();

    try request.sendBodiless();
    const response = try request.receiveHead(&.{});

    if (response.head.status != .ok) return error.RequestFailed;

    const content_length = response.head.content_length;
    const reader = request.reader.bodyReader(&transfer_buffer, .none, content_length);

    var body = std.ArrayList(u8).empty;
    errdefer body.deinit(allocator);

    var done = false;
    var bytes_read: usize = 0;
    while (!done) {
        const size = reader.readSliceShort(&reader_buffer) catch break;
        if (size > 0) {
            bytes_read += size;
            try body.appendSlice(allocator, reader_buffer[0..size]);
        }
        if (content_length) |c_len| {
            if (bytes_read >= c_len) done = true;
        }
        if (size < reader_buffer.len) done = true;
    }

    const parsed = try std.json.parseFromSlice(struct { tag_name: []const u8 }, allocator, body.items, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();

    return allocator.dupe(u8, parsed.value.tag_name);
}

pub fn runUpdateCommand(writer: *std.Io.Writer, allocator: std.mem.Allocator) !void {
    try writer.print("Fetching latest changes from GitHub...\n", .{});

    var git_proc = std.process.Child.init(&.{ "git", "pull", "origin", "main" }, allocator);
    _ = try git_proc.spawnAndWait();

    try writer.print("Building latest version...\n", .{});
    var build_proc = std.process.Child.init(&.{ "zig", "build", "-Doptimize=ReleaseFast" }, allocator);
    _ = try build_proc.spawnAndWait();

    try writer.print("\n\x1b[32mSuccessfully updated ziggy-llm to the latest version!\x1b[0m\n", .{});
    try writer.print("Note: The newly compiled binary is in zig-out/bin/ziggy-llm\n", .{});
}
