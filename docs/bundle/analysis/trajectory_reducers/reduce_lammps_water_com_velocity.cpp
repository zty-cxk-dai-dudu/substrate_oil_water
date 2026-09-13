#include <zlib.h>

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct AtomMap {
    int water = -1;
    double coefficient = 0.0;
};

static inline void skip_space(const char *&p) {
    while (*p == ' ' || *p == '\t') ++p;
}

static inline int parse_int(const char *&p) {
    skip_space(p);
    int sign = 1;
    if (*p == '-') { sign = -1; ++p; }
    int value = 0;
    while (*p >= '0' && *p <= '9') value = value * 10 + (*p++ - '0');
    return sign * value;
}

static inline double parse_double(const char *&p) {
    skip_space(p);
    double sign = 1.0;
    if (*p == '-') { sign = -1.0; ++p; }
    else if (*p == '+') { ++p; }
    double value = 0.0;
    while (*p >= '0' && *p <= '9') value = value * 10.0 + (*p++ - '0');
    if (*p == '.') {
        ++p;
        double scale = 0.1;
        while (*p >= '0' && *p <= '9') {
            value += scale * (*p++ - '0');
            scale *= 0.1;
        }
    }
    if (*p == 'e' || *p == 'E') {
        ++p;
        int exponent_sign = 1;
        if (*p == '-') { exponent_sign = -1; ++p; }
        else if (*p == '+') { ++p; }
        int exponent = 0;
        while (*p >= '0' && *p <= '9') exponent = exponent * 10 + (*p++ - '0');
        value *= std::pow(10.0, exponent_sign * exponent);
    }
    return sign * value;
}

static bool read_line(gzFile input, char *buffer, int size) {
    return gzgets(input, buffer, size) != nullptr;
}

static void require_line(gzFile input, char *buffer, int size, const char *context) {
    if (!read_line(input, buffer, size)) {
        std::cerr << "Unexpected EOF while reading " << context << "\n";
        std::exit(2);
    }
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " mapping.txt velocities.lammpstrj.gz output.bin\n";
        return 2;
    }
    const std::string mapping_path = argv[1];
    const std::string input_path = argv[2];
    const std::string output_path = argv[3];
    const std::string partial_path = output_path + ".partial";

    std::ifstream mapping_stream(mapping_path);
    int natoms = 0, nwater = 0, expected_frames = 0;
    mapping_stream >> natoms >> nwater >> expected_frames;
    if (!mapping_stream || natoms <= 0 || nwater <= 0 || expected_frames <= 0) {
        std::cerr << "Invalid mapping header\n";
        return 2;
    }
    std::vector<AtomMap> mapping(natoms + 1);
    int atom_id = 0, water = 0;
    double coefficient = 0.0;
    int mapped_atoms = 0;
    while (mapping_stream >> atom_id >> water >> coefficient) {
        if (atom_id < 1 || atom_id > natoms || water < 0 || water >= nwater) {
            std::cerr << "Invalid mapping row\n";
            return 2;
        }
        mapping[atom_id] = {water, coefficient};
        ++mapped_atoms;
    }
    if (mapped_atoms != 3 * nwater) {
        std::cerr << "Expected " << 3 * nwater << " mapped atoms, got "
                  << mapped_atoms << "\n";
        return 2;
    }

    gzFile input = gzopen(input_path.c_str(), "rb");
    FILE *output = std::fopen(partial_path.c_str(), "wb");
    if (!input || !output) {
        std::cerr << "Cannot open input or output: " << std::strerror(errno) << "\n";
        return 2;
    }

    constexpr int kBufferSize = 1024;
    char buffer[kBufferSize];
    int frames = 0;
    long long last_step = -1;
    std::vector<double> accumulated(nwater * 3, 0.0);
    std::vector<float> record(nwater * 3, 0.0f);
    while (read_line(input, buffer, kBufferSize)) {
        if (std::strncmp(buffer, "ITEM: TIMESTEP", 14) != 0) continue;
        require_line(input, buffer, kBufferSize, "timestep");
        const long long step = std::strtoll(buffer, nullptr, 10);
        require_line(input, buffer, kBufferSize, "atom-count header");
        require_line(input, buffer, kBufferSize, "atom count");
        const int frame_atoms = std::atoi(buffer);
        if (frame_atoms != natoms) {
            std::cerr << "Unexpected atom count at step " << step << "\n";
            return 2;
        }
        require_line(input, buffer, kBufferSize, "box header");
        for (int axis = 0; axis < 3; ++axis)
            require_line(input, buffer, kBufferSize, "box bounds");
        require_line(input, buffer, kBufferSize, "atom header");
        if (std::strstr(buffer, " vx ") == nullptr || std::strstr(buffer, " vz") == nullptr) {
            std::cerr << "Velocity columns not found\n";
            return 2;
        }
        std::fill(accumulated.begin(), accumulated.end(), 0.0);
        for (int row = 0; row < natoms; ++row) {
            require_line(input, buffer, kBufferSize, "atom row");
            const char *p = buffer;
            const int id = parse_int(p);
            if (id < 1 || id > natoms || mapping[id].water < 0) continue;
            (void)parse_int(p);
            skip_space(p);
            while (*p && *p != ' ' && *p != '\t' && *p != '\n') ++p;
            const double velocity[3] = {
                parse_double(p), parse_double(p), parse_double(p)
            };
            const AtomMap &entry = mapping[id];
            for (int axis = 0; axis < 3; ++axis)
                accumulated[entry.water * 3 + axis] += entry.coefficient * velocity[axis];
        }
        if (step == 0) continue;
        const long long expected_step = 2LL * (frames + 1);
        if (step != expected_step) {
            std::cerr << "Expected step " << expected_step << ", got " << step << "\n";
            return 2;
        }
        for (size_t i = 0; i < record.size(); ++i)
            record[i] = static_cast<float>(accumulated[i]);
        if (std::fwrite(record.data(), sizeof(float), record.size(), output)
            != record.size()) {
            std::cerr << "Write failed\n";
            return 2;
        }
        ++frames;
        last_step = step;
        if (frames % 10000 == 0) {
            std::cout << "{\"velocity_frames_done\":" << frames
                      << ",\"velocity_frames_total\":" << expected_frames << "}\n";
            std::cout.flush();
        }
    }
    gzclose(input);
    if (std::fclose(output) != 0) {
        std::cerr << "Failed closing output\n";
        return 2;
    }
    if (frames != expected_frames || last_step != 2LL * expected_frames) {
        std::cerr << "Incomplete dump: frames=" << frames << ", last_step="
                  << last_step << "\n";
        return 2;
    }
    if (std::rename(partial_path.c_str(), output_path.c_str()) != 0) {
        std::cerr << "Cannot finalize output: " << std::strerror(errno) << "\n";
        return 2;
    }
    std::cout << "{\"status\":\"complete\",\"mapped_atoms\":" << mapped_atoms
              << ",\"frames\":" << frames << ",\"waters\":" << nwater
              << ",\"last_step\":" << last_step
              << ",\"velocity_unit\":\"A/fs\"}\n";
    return 0;
}
