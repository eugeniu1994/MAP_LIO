#pragma once

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

namespace timer {
typedef std::chrono::_V2::system_clock::time_point clocktime;

struct measurement {
  std::string id;
  std::vector<double> times;
  clocktime t_start;
};

struct descr {
  double mean, median, max;
  int n;
};

inline std::vector<struct measurement> measurements;

inline clocktime tic() { return std::chrono::high_resolution_clock::now(); }

inline double clock_to_ms(const clocktime t_start, const clocktime t_end) {
  std::chrono::duration<double, std::milli> ms_dur = t_end - t_start;
  return ms_dur.count();
}

inline size_t get_measurement_index(const char *id) {
  const std::string sid(id);
  for (size_t i = 0; i < measurements.size(); ++i) {
    if (measurements[i].id == sid) {
      return i;
    }
  }

  struct measurement m = {};
  m.id = sid;
  measurements.push_back(m);
  return measurements.size() - 1;
}

inline void start(const char *id) {
  const size_t i = get_measurement_index(id);
  measurements[i].t_start = tic();
}

inline void end(const char *id) {
  const clocktime t_end = tic();
  const size_t i = get_measurement_index(id);
  struct measurement &m = measurements[i];
  m.times.push_back(clock_to_ms(m.t_start, t_end));
}

inline struct descr get_descr(std::vector<double> &arr) {
  struct descr descr = {};
  const size_t n = arr.size();
  if (n == 0) {
    return descr;
  }

  std::sort(arr.begin(), arr.end());
  double sum = 0.0;
  for (double v : arr) {
    sum += v;
  }
  descr.mean = sum / n;

  if (n & 1) {
    descr.median = arr[n / 2];
  } else {
    descr.median = (arr[n / 2] + arr[n / 2 + 1]) / 2.0;
  }

  descr.max = arr[n - 1];
  descr.n = (int)n;
  return descr;
}

inline void print(const char *filename = NULL) {
  FILE *file = NULL;
  if (filename != NULL) {
    file = fopen(filename, "w");
    if (file == NULL) {
      (void)fprintf(stderr, "failed to open %s\n", filename);
    } else {
      (void)fprintf(file, "id,count,mean,median,max\n");
    }
  }

  for (struct measurement &m : measurements) {
    const struct descr descr = get_descr(m.times);

    if (file == NULL) {
      (void)printf(
          "%20s (N = %6i): mean = %7.2lf, median = %7.2lf, max = %7.2lf (ms)\n",
          m.id.c_str(), descr.n, descr.mean, descr.median, descr.max);
    } else {
      (void)fprintf(file, "%s,%i,%lf,%lf,%lf\n", m.id.c_str(), descr.n,
                    descr.mean, descr.median, descr.max);
    }
  }

  if (file != NULL) {
    (void)fclose(file);
  }
}
} // namespace timer
