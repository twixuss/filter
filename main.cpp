#define _CRT_SECURE_NO_WARNINGS

#define TL_IMPL
#include <tl/math.h>
#include <tl/console.h>
#include <tl/main.h>
#include <tl/file.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <algorithm>

using namespace tl;

using Pixel = v4u8;

template <class A, class B>
inline void dilate(Pixel *source_pixels, Pixel *destination_pixels, v2s size, s32 radius, A &&should_be_dilated, B &&get_length)
    requires requires { !!should_be_dilated(Pixel{}); }
{
    print("Building offset table...\n");

    List<v2s> offsets;
    defer { free(offsets); };

    offsets.reserve(size.x*size.y);

    for (s32 iy = 0; iy < size.y; ++iy) {
    for (s32 ix = 0; ix < size.x; ++ix) {
        auto offset = v2s{ix,iy} - size/2;
        if (get_length(offset) <= radius)
            offsets.add(offset);
    }
    }

    std::sort(offsets.begin(), offsets.end(), [&](v2s a, v2s b) {
        return get_length(a) < get_length(b);
    });

    for (s32 iy = 0; iy < size.y; ++iy) {
    print("Row {}\n", iy);
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

struct Filter {
    Span<utf8> name;
    bool (*parse)(Span<Span<utf8>> options, void *_state);
    v2s (*get_destination_size)(v2s source_size, void *_state);
    bool (*apply)(Pixel *source_pixels, v2s source_size, Pixel *destination_pixels, v2s destination_size, void *_state);
};

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
    else if (selected_options[i] == u8#name##s) { \
        ++i; \
        if (i >= selected_options.count) { \
            with(ConsoleColor::red, print("Error: ")); \
            print("Expected an integer after '{}', but got nothing\n", u8#name##s); \
            return false; \
        } \
        auto parsed = parse_u64(selected_options[i]); \
        if (!parse_option(u8#name##s, selected_options[i], &state.name)) { \
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
            with(ConsoleColor::red, print("Error: ")); \
            print("Option '{}' not found\n", selected_options[i]); \
            return false; \
        } \
    }

#define _DEFINE_ENUM_VALUE(name) name,
#define _DEFINE_ENUM_NAME(name) u8#name##s,

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

List<Filter> filters;
s32 tl_main(Span<Span<utf8>> args) {
    init_printer();

    construct(filters);

    {
        #define ENUMERATE_OPTIONS(e) \
            e(s32, radius, 0) \
            e(s32, threshold, 128) \
            e(DistanceMethod, distance, {}) \

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

                print("dilate\n");
                print("radius: {}\n", radius);
                print("threshold: {}\n", threshold);
                print("distance: {}\n", distance);

                // NOTE: this makes inlining possible
                switch (distance.value) {
                    case DistanceMethod::euclidean: dilate(source_pixels, destination_pixels, source_size, radius, [&](Pixel p){ return p.w < threshold; }, [&](v2s b){ return length(b); }); break;
                    case DistanceMethod::chebyshev: dilate(source_pixels, destination_pixels, source_size, radius, [&](Pixel p){ return p.w < threshold; }, [&](v2s b){ return max(absolute(b)); }); break;
                    case DistanceMethod::manhattan: dilate(source_pixels, destination_pixels, source_size, radius, [&](Pixel p){ return p.w < threshold; }, [&](v2s b){ return sum(absolute(b)); }); break;
                }

                return true;
            },
        });

        #undef ENUMERATE_OPTIONS
    }

    if (args.count < 4) {
        print(R"(Usage: {} <input> <output> <filter> [<filter options>]
Filters
)", args[0]);

        for (auto &filter : filters) {
            print("  {}\n", filter.name);
        }
        return 1;
    }

    auto input_path = args[1];
    auto output_path = args[2];
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
