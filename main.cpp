#define _CRT_SECURE_NO_WARNINGS

#define TL_IMPL
#include <tl/math.h>
#include <tl/console.h>
#include <tl/main.h>
#include <tl/file.h>
#include <tl/time.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

using namespace tl;

using Pixel = v4u8;

template <class A, class B>
inline void dilate(Pixel *source_pixels, Pixel *destination_pixels, v2s size, s32 radius, bool smooth, A &&should_be_dilated, B &&get_length)
    requires requires { {should_be_dilated(Pixel{}) } -> std::same_as<bool>; }
{
    print("Building offset table...\n");

    List<v2s16> offsets;
    defer { free(offsets); };

    offsets.reserve(size.x*size.y);

    for (s32 iy = 0; iy < size.y; ++iy) {
    for (s32 ix = 0; ix < size.x; ++ix) {
        auto offset = v2s16{(s16)ix,(s16)iy} - (v2s16)size/2;
        if (get_length(offset) <= radius)
            offsets.add(offset);
    }
    }

    quick_sort(offsets, get_length);
    
    if (smooth) {
        for (s32 iy = 0; iy < size.y; ++iy) {
            print("\rRow {}", iy);
            for (s32 ix = 0; ix < size.x; ++ix) {

                Pixel p = source_pixels[iy*size.x + ix];

                if (should_be_dilated(p)) {
                    struct FactoredPixel {
                        Pixel pixel;
                        f32 factor;
                    };
                    scoped(temporary_storage_checkpoint);
                    List<FactoredPixel, TemporaryAllocator> closest_pixels;
                    f32 closest_pixel_distance = 0;
                    for (auto offset : offsets/*.skip(next_time_starting_from)*/) {
                        smm jx = ix + offset.x;
                        smm jy = iy + offset.y;

                        if ((umm)jx >= size.x) continue;
                        if ((umm)jy >= size.y) continue;

                        auto t = source_pixels[jy*size.x + jx];
                        if (!should_be_dilated(t)) {
                            f32 distance = length(offset);

                            //f32 const max_distance = sqrt2 - 1;
                            f32 const max_distance = 1;

                            if (closest_pixels.count == 0) {
                                closest_pixel_distance = distance;
                                closest_pixels.add({t, 1});
                            } else {
                                if (distance >= closest_pixel_distance + max_distance) {
                                    //update_starting_index(index_of(offsets, &offset), 2);
                                    break;
                                }
                                //closest_pixels.add({t, (distance - closest_pixel_distance) / max_distance});
                                closest_pixels.add({t, 1});
                            }
                        }
                    }

                    v3f color_sum = {};
                    f32 factor_sum = {};
                    for (auto t : closest_pixels) {
                        color_sum += (v3f)t.pixel.xyz * t.factor;
                        factor_sum += t.factor;
                    }

                    p.xyz = autocast (color_sum / factor_sum);
                    p.w = 255;
                } else {
                    p.w = 255;
                }

                destination_pixels[iy*size.x + ix] = p;
            }
        }
    } else {
        for (s32 iy = 0; iy < size.y; ++iy) {
            print("\rRow {}", iy);
            for (s32 ix = 0; ix < size.x; ++ix) {

                Pixel p = source_pixels[iy*size.x + ix];

                if (should_be_dilated(p)) {
                    for (auto offset : offsets) {
                        u32 jx = ix + offset.x;
                        u32 jy = iy + offset.y;

                        if (jx >= (u32)size.x) continue;
                        if (jy >= (u32)size.y) continue;

                        auto t = source_pixels[jy*size.x + jx];
                        if (!should_be_dilated(t)) {
                            p.xyz = t.xyz;
                            p.w = 255;
                            break;
                        }
                    }
                } else {
                    p.w = 255;
                }

                destination_pixels[iy*size.x + ix] = p;
            }
        }
    }
}

struct Filter {
    Span<utf8> name;
    bool (*parse)(Span<Span<utf8>> options, void *_state);
    v2s (*get_destination_size)(v2s source_size, void *_state);
    bool (*apply)(Pixel *source_pixels, v2s source_size, Pixel *destination_pixels, v2s destination_size, void *_state);
};

bool parse_option(Span<utf8> name, Span<utf8> value, bool *result) {
    auto is_true = [](Span<utf8> value) {
        if (equals_case_insensitive(value, u8"true"s)) return true;
        if (value == u8"1"s) return true;
        if (value == u8"yes"s) return true;
        return false;
    };

    auto is_false = [](Span<utf8> value) {
        if (equals_case_insensitive(value, u8"false"s)) return true;
        if (value == u8"0"s) return true;
        if (value == u8"no"s) return true;
        return false;
    };

    if (is_true (value)) { *result = true; return true; }
    if (is_false(value)) { *result = false; return true; }
   
    with(ConsoleColor::red, print("Error: "));
    print("Expected a boolean after '{}', but got '{}'n", name, value);
    return false;
}

bool parse_option(Span<utf8> name, Span<utf8> value, s32 *result) {
    auto parsed = parse_u64(value);
    if (!parsed) {
        with(ConsoleColor::red, print("Error: "));
        print("Expected an integer after '{}', but got '{}'n", name, value);
        return false;
    }

    *result = parsed.value_unchecked();
    return true;
}

#include <string>
#include <stdexcept>

bool parse_option(Span<utf8> name, Span<utf8> value, f32 *result) {
    try {
        *result = std::stof(std::string{(char *)value.begin(), (char *)value.end()});
        return true;
    } catch(std::invalid_argument&) {
        with(ConsoleColor::red, print("Error: "));
        print("Expected a float after '{}', but got '{}'n", name, value);
        return false;
    }
}

template <class T>
bool parse_option(Span<utf8> name, Span<utf8> value, T *result)
    requires requires { T::is_usable_enum == true; }
{
    for (umm i = 0; i < count_of(T::names); ++i) {
        if (value == T::names[i]) {
            result->value = (decltype(result->value)) i;
            return true;
        }
    }
    with(ConsoleColor::red, print("Error: "));
    print("Expected one of following values after '{}', but got '{}'n", name, value);
    for (auto name : T::names) {
        print("  {}\n", name);
    }
    return false;
}

#define _DEFINE_MEMBER(type, name, default) type name = default;
#define _PARSE_OPTION(type, name, default) \
    else if (selected_options[i] == u8###name##s) { \
        ++i; \
        if (i >= selected_options.count) { \
            with(ConsoleColor::red, print("Error: ")); \
            print("Expected an integer after '{}', but got nothing\n", u8###name##s); \
            return false; \
        } \
        auto parsed = parse_u64(selected_options[i]); \
        if (!parse_option(u8###name##s, selected_options[i], &state.name)) { \
            return false; \
        } \
    }
#define DEFINE_OPTIONS \
    struct Options { \
        ENUMERATE_OPTIONS(_DEFINE_MEMBER) \
    };

#define DEFINE_STATE \
    auto &state = *(Options *)_state; \

#define PARSE_OPTIONS \
    for (umm i = 0; i < selected_options.count; ++i) { \
        if (false) {} \
        ENUMERATE_OPTIONS(_PARSE_OPTION) \
        else { \
            with(ConsoleColor::red, print("Warning: ")); \
            print("Option '{}' not found, ignoring\n", selected_options[i]); \
        } \
    }

#define _DEFINE_ENUM_VALUE(name) name,
#define _DEFINE_ENUM_NAME(name) u8###name##s,

#define DEFINE_ENUM(enum_name) \
    struct enum_name { \
        inline static constexpr bool is_usable_enum = true; \
        enum { \
            ENUMERATE_ENUM(_DEFINE_ENUM_VALUE) \
        } value; \
        inline static constexpr Span<utf8> names[] { \
            ENUMERATE_ENUM(_DEFINE_ENUM_NAME) \
        }; \
    }; \
    inline umm append(StringBuilder &builder, enum_name e) { \
        return append(builder, e.names[(umm)e.value]); \
    }


#define ENUMERATE_ENUM(e) \
    e(euclidean) \
    e(manhattan) \
    e(chebyshev) \

DEFINE_ENUM(DistanceMethod);

#undef ENUMERATE_ENUM

#define ENUMERATE_ENUM(e) \
    e(average) \
    e(sum) \

DEFINE_ENUM(Blend);

#undef ENUMERATE_ENUM

List<Filter> filters;
s32 tl_main(Span<Span<utf8>> args) {
    init_printer();

    construct(filters);

    {
        #define ENUMERATE_OPTIONS(e) \
            e(s32, radius, 0) \
            e(s32, threshold, 128) \
            e(DistanceMethod, distance, {}) \
            e(bool, smooth, true) \

        DEFINE_OPTIONS;

        filters.add({
            .name = u8"dilate"s,
            .parse = [](Span<Span<utf8>> selected_options, void *_state) -> bool {
                DEFINE_STATE;
                state = {};

                PARSE_OPTIONS;

                return true;
            },
            .get_destination_size = [](v2s source_size, void *_state) -> v2s {
                return source_size;
            },
            .apply = [](Pixel *source_pixels, v2s source_size, Pixel *destination_pixels, v2s destination_size, void *_state) -> bool {
                DEFINE_STATE;


                auto radius = state.radius ? state.radius : max(source_size);
                auto threshold = state.threshold;
                auto distance = state.distance;
                auto smooth = state.smooth;

                print("radius: {}\n", radius);
                print("threshold: {}\n", threshold);
                print("distance: {}\n", distance);
                print("smooth: {}\n", smooth);

                switch (distance.value) {
                    case DistanceMethod::euclidean: dilate(source_pixels, destination_pixels, source_size, (f32)radius, smooth, [&](Pixel p){ return p.w < threshold; }, [&](auto b){ return length(b); }); break;
                    case DistanceMethod::chebyshev: dilate(source_pixels, destination_pixels, source_size, (f32)radius, smooth, [&](Pixel p){ return p.w < threshold; }, [&](auto b){ return max(absolute(b)); }); break;
                    case DistanceMethod::manhattan: dilate(source_pixels, destination_pixels, source_size, (f32)radius, smooth, [&](Pixel p){ return p.w < threshold; }, [&](auto b){ return sum(absolute(b)); }); break;
                }

                return true;
            },
        });

        #undef ENUMERATE_OPTIONS
    }

    {
        #define ENUMERATE_OPTIONS(e) \
            e(s32, radius, 8) \
            e(s32, percent, 50) \

        DEFINE_OPTIONS;

        filters.add({
            .name = u8"median"s,
            .parse = [](Span<Span<utf8>> selected_options, void *_state) -> bool {
                DEFINE_STATE;
                state = {};

                PARSE_OPTIONS;

                return true;
            },
            .get_destination_size = [](v2s source_size, void *_state) -> v2s {
                return source_size;
            },
            .apply = [](Pixel *source_pixels, v2s source_size, Pixel *destination_pixels, v2s destination_size, void *_state) -> bool {
                DEFINE_STATE;


                auto radius = state.radius;
                auto percent = clamp(state.percent, 0, 100);

                print("radius: {}\n", radius);
                print("percent: {}\n", percent);

                auto diameter = (umm)radius*2 + 1;

                Span<Pixel> buf = { current_allocator.allocate<Pixel>(diameter*diameter), (umm)0 };
                defer { current_allocator.free(buf.data); };

                for (smm py = 0; py < source_size.y; ++py) {
                print("Row {}\r", py);
                for (smm px = 0; px < source_size.x; ++px) {

                    buf.count = 0;

                    for (smm oy = -radius; oy <= +radius; ++oy) {
                    for (smm ox = -radius; ox <= +radius; ++ox) {

                        smm x = px + ox;
                        smm y = py + oy;

                        x = frac(x, (smm)source_size.x);
                        y = frac(y, (smm)source_size.y);

                        if (ox*ox + oy*oy <= radius*radius) {
                            buf.data[buf.count++] = source_pixels[y*source_size.x + x];
                        }
                    }
                    }

                    quick_sort(buf, [](Pixel a) { return dot((v3f)a.xyz, v3f{0.299f, 0.587f, 0.114f}); });

                    destination_pixels[py*destination_size.x + px] = buf[min(buf.count * percent / 100, buf.count - 1)];
                }
                }

                return true;
            },
        });

        #undef ENUMERATE_OPTIONS
    }

    {
        #define ENUMERATE_OPTIONS(e) \
            e(s32, radius, 8) \
            e(f32, scale, 1) \

        DEFINE_OPTIONS;

        filters.add({
            .name = u8"bilateral"s,
            .parse = [](Span<Span<utf8>> selected_options, void *_state) -> bool {
                DEFINE_STATE;
                state = {};

                PARSE_OPTIONS;

                return true;
            },
            .get_destination_size = [](v2s source_size, void *_state) -> v2s {
                return source_size;
            },
            .apply = [](Pixel *source_pixels, v2s source_size, Pixel *destination_pixels, v2s destination_size, void *_state) -> bool {
                DEFINE_STATE;


                auto radius = state.radius;
                auto scale = clamp(state.scale, 0.f, 10.f);

                print("radius: {}\n", radius);
                print("scale: {}\n", scale);

                for (smm py = 0; py < source_size.y; ++py) {
                print("Row {}\r", py);
                for (smm px = 0; px < source_size.x; ++px) {
                    v4f c = (v4f)source_pixels[py*source_size.x + px];

                    v4f sum = {};
                    f32 den = 0;

                    auto weight = [&] (v4f a, v4f b) -> f32 {
                        return 1.f - clamp(manhattan(a, b)/(255*3) * scale, 0.f, 1.f);
                    };

                    for (smm oy = -radius; oy <= +radius; ++oy) {
                    for (smm ox = -radius; ox <= +radius; ++ox) {

                        smm x = px + ox;
                        smm y = py + oy;

                        x = frac(x, (smm)source_size.x);
                        y = frac(y, (smm)source_size.y);

                        if (ox*ox + oy*oy <= radius*radius) {
                            auto p = (v4f)source_pixels[y*source_size.x + x];
                            auto w = weight(c, p);
                            sum += (v4f)p * w;
                            den += w;
                            //buf.data[buf.count++] = source_pixels[y*source_size.x + x];
                        }
                    }
                    }

                    destination_pixels[py*destination_size.x + px] = (Pixel)(sum / den);
                }
                }

                return true;
            },
        });

        #undef ENUMERATE_OPTIONS
    }

    {
        #define ENUMERATE_OPTIONS(e) \
            e(s32, radius, 8) \

        DEFINE_OPTIONS;

        filters.add({
            .name = u8"kuwahara"s,
            .parse = [](Span<Span<utf8>> selected_options, void *_state) -> bool {
                DEFINE_STATE;
                state = {};

                PARSE_OPTIONS;

                return true;
            },
            .get_destination_size = [](v2s source_size, void *_state) -> v2s {
                return source_size;
            },
            .apply = [](Pixel *source_pixels, v2s source_size, Pixel *destination_pixels, v2s destination_size, void *_state) -> bool {
                DEFINE_STATE;


                auto radius = state.radius;

                print("radius: {}\n", radius);

                auto quadrant_width = radius+1;
                auto quadrant_area = pow2(quadrant_width);

                Span<Pixel> buf = { current_allocator.allocate<Pixel>(quadrant_area*4), (umm)0 };
                defer { current_allocator.free(buf.data); };

                struct Quadrant {
                    Span<Pixel> pixels;
                    v2s offset;
                    f32 stddev;
                };

                Quadrant quadrants[4] = {
                    { buf.subspan(quadrant_area*0, quadrant_area), { -radius, -radius} },
                    { buf.subspan(quadrant_area*1, quadrant_area), {       0, -radius} },
                    { buf.subspan(quadrant_area*2, quadrant_area), { -radius,       0} },
                    { buf.subspan(quadrant_area*3, quadrant_area), {       0,       0} },
                };

                for (s32 py = 0; py < source_size.y; ++py) {
                print("Row {}\r", py);
                for (s32 px = 0; px < source_size.x; ++px) {
                    auto c = source_pixels[py*source_size.x + px];

                    for (auto &q : quadrants) {
                        for (s32 oy = 0; oy <= radius; ++oy) {
                        for (s32 ox = 0; ox <= radius; ++ox) {
                            s32 sx = frac(px + ox + q.offset.x, source_size.x);
                            s32 sy = frac(py + oy + q.offset.y, source_size.y);
                            q.pixels.data[oy*quadrant_width + ox] = source_pixels[sy*source_size.x + sx];
                        }
                        }

                        v4f avg = {};
                        for (auto p : q.pixels) {
                            avg += (v4f)p;
                        }
                        avg /= q.pixels.count;

                        v4f stddev = {};
                        for (auto p : q.pixels) {
                            stddev += pow2((v4f)p - avg);
                        }

                        stddev = sqrt(stddev / q.pixels.count);

                        q.stddev = length_squared(stddev);
                    }

                    Quadrant *min_q = &quadrants[0];
                    for (auto &q : quadrants) {
                        if (q.stddev < min_q->stddev)
                            min_q = &q;
                    }

                    v4s avg = {};
                    for (auto &p : min_q->pixels)
                        avg += (v4s)p;
                    avg /= min_q->pixels.count;

                    destination_pixels[py*destination_size.x + px] = (Pixel)avg;
                }
                }

                return true;
            },
        });

        #undef ENUMERATE_OPTIONS
    }

    {
        #define ENUMERATE_OPTIONS(e) \
            e(s32, slices, 36) \
            e(Blend, blend, {Blend::average}) \

        DEFINE_OPTIONS;

        filters.add({
            .name = u8"skidmark"s,
            .parse = [](Span<Span<utf8>> selected_options, void *_state) -> bool {
                DEFINE_STATE;
                state = {};

                PARSE_OPTIONS;

                return true;
            },
            .get_destination_size = [](v2s source_size, void *_state) -> v2s {
                DEFINE_STATE;
                return {source_size.x, state.slices};
            },
            .apply = [](Pixel *source_pixels, v2s source_size, Pixel *destination_pixels, v2s destination_size, void *_state) -> bool {
                DEFINE_STATE;

                for (int slice = 0; slice < state.slices; ++slice) {
                    print("slice {}\r", slice);
                    // NOTE: do only 180 degrees, because two halfs are identical
                    float angle = (float)slice / state.slices * pi;

                    switch (state.blend.value) {
                        case Blend::average:
                        case Blend::sum: {
                            for (int x = 0; x < source_size.x; ++x) {
                                v4u32 column_sum = {};
                                for (int y = 0; y < source_size.y; ++y) {
                                    v2f half_size = (v2f)source_size * 0.5f;
                                    v2f pf = (v2f)v2s{x, y};
                                    pf -= half_size;

                                    if (length(pf) > source_size.x * 0.5f)
                                        continue;

                                    pf = m2::rotation(angle) * pf;
                                    pf += half_size;

                                    v2s p = round_to_int(pf);

                                    if (p.x >= 0 && p.y >= 0 && p.x < source_size.x && p.y < source_size.y) {
                                        column_sum += (v4u32)source_pixels[p.y*source_size.x + p.x];
                                    }
                                }
                                if (state.blend.value == Blend::average)
                                    column_sum /= source_size.y;
                                column_sum = clamp(column_sum, (v4u32)V4s(0), (v4u32)V4s(255));
                                destination_pixels[slice*destination_size.x + x] = (Pixel)column_sum;
                            }
                            break;
                        }
                    }
                }

                return true;
            },
        });

        #undef ENUMERATE_OPTIONS
    }

    if (args.count < 4) {
        print(R"(Usage: {} <input> (<output>|-i) <filter> [<filter options>]
Filters
)", args[0]);

        for (auto &filter : filters) {
            print("  {}\n", filter.name);
        }
        return 1;
    }

    auto input_path = args[1];
    auto output_path = args[2];
    if (output_path == u8"-i"s)
        output_path = input_path;

    auto filter_name = args[3];

    auto found_filter = find_if(filters, [&](auto filter){return filter.name == filter_name;});
    if (!found_filter) {
        with(ConsoleColor::red, print("Error: "));
        print("Filter '{}' not found\n", filter_name);
        return 2;
    }

    auto filter = *found_filter;

    u8 filter_state[1024*1024];
    if (!filter.parse(args.skip(4), filter_state)) {
        return 3;
    }

    auto input_buffer = read_entire_file(input_path);
    if (!input_buffer.data) {
        with(ConsoleColor::red, print("Error: "));
        print("Failed to read '{}'\n", input_path);
        return 4;
    }
    defer { free(input_buffer); };

    v2s source_size;
    auto source_pixels = (Pixel *)stbi_load_from_memory(input_buffer.data, input_buffer.count, &source_size.x, &source_size.y, 0, 4);
    if (!source_pixels) {
        with(ConsoleColor::red, print("Error: "));
        print("Failed to decode '{}'\n", input_path);
        return 5;
    }
    defer { stbi_image_free(source_pixels); };

    auto destination_size = filter.get_destination_size(source_size, filter_state);

    auto destination_pixels = current_allocator.allocate<Pixel>(destination_size.x*destination_size.y);

    if (!filter.apply(source_pixels, source_size, destination_pixels, destination_size, filter_state)) {
        return 6;

    }

    if (!stbi_write_png((char *)output_path.data, destination_size.x, destination_size.y, 4, destination_pixels, sizeof(destination_pixels[0]) * destination_size.x)) {
        with(ConsoleColor::red, print("Error: "));
        print("Failed to write '{}'\n", output_path);
        return 7;
    }

    return 0;
}
