#include <iostream>
#include <random>
#include <algorithm>
#include <fstream>
#include <climits>
#include <set>
#include <vector>
#include <cstdio>
#include <cstdlib>

int main()
{
  double scale = 1e+6;
  double max = double(INT_MAX) / scale;
  int nelements = 190;

  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::lognormal_distribution<double> dist(0.0, 2.0);

  std::set<int> samples;

  while (samples.size() < nelements) {
    double r = dist(rng);
    if (r > max) continue;
    samples.insert(int(r * scale));
  }

  std::vector<int> vec(samples.begin(), samples.end());
  std::sort(vec.begin(), vec.end());

  FILE *fout = fopen("lognormal.sorted.190.txt", "w");
  for (int x : vec) {
    fprintf(fout, "%d\n", x);
  }
  fclose(fout);

  return 0;
}