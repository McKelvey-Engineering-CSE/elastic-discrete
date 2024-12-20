/*
Runs regression tests.
Args:
* path to clustering launcher
* Path to testcase directory
* Additional args to gtest
*/

#include <gtest/gtest.h>
#include "subprocess.h"
#include <unordered_map>
#include <stdio.h>
#include <filesystem>
#include <unistd.h>
#include <fstream>

/* test utilities */

typedef std::unordered_map<std::string, int> core_map;

std::filesystem::path clustering_launcher_path;
std::filesystem::path testcase_dir;

struct clusteringRunResult {
  int exit_code;
  std::string stdout;
  std::string stderr;
};

core_map parse_core_assignments (const std::string & stdout) {
  core_map assignments;

  std::istringstream stream(stdout);
  std::string line;
  while (getline(stream, line)) {
    int taskidx;
    int taskiter;
    int core_count;
    if (sscanf(line.c_str(), "TEST: [%d,%d] core count: %d", &taskidx, &taskiter, &core_count) == 3) {
      assignments.insert({std::to_string(taskidx) + "," + std::to_string(taskiter), core_count});
    }
  }

  return assignments;
}

struct clusteringRunResult runClustering(const std::string test_name) {
  struct clusteringRunResult result;

  std::filesystem::path test_path = testcase_dir / (test_name + ".yaml");

  const char * command_line[] = {clustering_launcher_path.c_str(), test_path.c_str(), NULL};
  struct subprocess_s subprocess;

  int result_code = subprocess_create(command_line, 0, &subprocess);
  FILE* p_stdout = subprocess_stdout(&subprocess);
  FILE* p_stderr = subprocess_stderr(&subprocess);
  int process_return;
  int result2 = subprocess_join(&subprocess, &process_return);

  result.exit_code = process_return;
  if (process_return != 0) {
    return result;
  }

  std::string proc_stdout = "";
  std::string proc_stderr = "";

  int bufsize = 2048;
  char buffer[bufsize];

  while (fgets(buffer, bufsize, p_stdout) != nullptr) {
        proc_stdout += std::string(buffer, strlen(buffer)); // avoid inserting null terminators into the string
  }

  while (fgets(buffer, bufsize, p_stderr) != nullptr) {
        proc_stderr += std::string(buffer, strlen(buffer));
  }

  result.stdout = proc_stdout;
  result.stderr = proc_stderr;

  return result;
}

std::string get_expected_output(const std::string test_name) {
    std::filesystem::path test_path = testcase_dir / (test_name + ".txt");
    std::ifstream f(test_path.generic_string());
    std::stringstream buffer;
    buffer << f.rdbuf();
    return buffer.str();
}

void assert_same_core_assignments(const core_map & actual, const core_map & expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (auto & assignment : actual) {
    auto expected_val = expected.find(assignment.first);
    ASSERT_EQ(expected_val == expected.end(), false); // not found
    ASSERT_EQ(assignment.second, expected_val->second);
  }
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    printf("too few arguments\n");
    return -1;
  }

  clustering_launcher_path = ((std::filesystem::path) "./") / ((std::filesystem::path) argv[1]);
  testcase_dir = ((std::filesystem::path) "./") / ((std::filesystem::path) argv[2]);

  argc -= 2;
  argv = &argv[2];

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

/* Begin tests */

TEST(Scheduler, BasicAssignment) {
    clusteringRunResult result = runClustering("basic");
    ASSERT_EQ(result.exit_code, 0);
    assert_same_core_assignments(parse_core_assignments(result.stdout), parse_core_assignments(get_expected_output("basic")));
}