#include "check.h"
#include "galactic_tides/params.h"

#include <stdexcept>

int main() {
  galactic_tides::Params p;
  p.load_string(
      "# comment\n"
      "theta = 0.6   # trailing comment\n"
      "name=galaxy\n"
      "n = 1000\n"
      "flag = yes\n");
  CHECK_CLOSE(p.get_double("theta", 0.0), 0.6, 1e-15);
  CHECK(p.get_string("name", "") == "galaxy");
  CHECK(p.get_int("n", 0) == 1000);
  CHECK(p.get_bool("flag", false));
  CHECK(p.get_int("missing", 7) == 7);

  auto rest = p.apply_overrides({"theta=0.4", "positional"});
  CHECK(rest.size() == 1 && rest[0] == "positional");
  CHECK_CLOSE(p.get_double("theta", 0.0), 0.4, 1e-15);

  p.set("typo_key", "1");
  auto unused = p.unused();
  CHECK(unused.size() == 1 && unused[0] == "typo_key");

  bool threw = false;
  try {
    p.get_double("name", 0.0);
  } catch (const std::runtime_error &) {
    threw = true;
  }
  CHECK(threw);
  return TEST_MAIN_RESULT();
}
