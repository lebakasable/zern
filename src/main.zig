const std = @import("std");

inline fn sigmoid(value: anytype) @TypeOf(value) {
    return 1 / (1 + std.math.exp(-value));
}

const train = [_][3]f64{
    .{ 0, 0, 0 },
    .{ 1, 0, 1 },
    .{ 0, 1, 1 },
    .{ 1, 1, 1 },
};

fn cost(w1: f64, w2: f64, b: f64) f64 {
    var result: f64 = 0;
    for (0.., train) |i, _| {
        const x1 = train[i][0];
        const x2 = train[i][1];
        const y = sigmoid(x1 * w1 + x2 * w2 + b);
        const d = y - train[i][2];
        result += d * d;
    }

    return result / train.len;
}

pub fn main() !void {
    var rng = std.rand.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        try std.os.getrandom(std.mem.asBytes(&seed));
        break :blk seed;
    });

    var w1 = rng.random().float(f64);
    var w2 = rng.random().float(f64);
    var b = rng.random().float(f64);

    const eps = 1e-1;
    const rate = 1e-1;

    for (0..1e5) |_| {
        const c = cost(w1, w2, b);
        const dw1 = (cost(w1 + eps, w2, b) - c) / eps;
        const dw2 = (cost(w1, w2 + eps, b) - c) / eps;
        const db = (cost(w1, w2, b + eps) - c) / eps;
        w1 -= rate * dw1;
        w2 -= rate * dw2;
        b -= rate * db;
    }

    var i: f64 = 0;
    while (i < 2) : (i += 1) {
        var j: f64 = 0;
        while (j < 2) : (j += 1) {
            std.debug.print("{d} | {d} = {d}\n", .{ i, j, sigmoid(i * w1 + j * w2 + b) });
        }
    }
}
