#pragma once

#include <map>
#include <string>
#include <vector>

namespace galactic_tides {

// Minimal "key = value" parameter store. Lines starting with '#' are comments,
// trailing '# ...' comments are stripped. Later definitions override earlier
// ones, so command line "key=value" overrides can be applied after a file.
class Params {
 public:
  void load_file(const std::string &path);
  void load_string(const std::string &text, const std::string &origin = "<string>");
  // Applies "key=value" tokens; returns any tokens that are not of that form.
  std::vector<std::string> apply_overrides(const std::vector<std::string> &tokens);

  void set(const std::string &key, const std::string &value) { values_[key] = value; }
  bool has(const std::string &key) const { return values_.count(key) != 0; }

  std::string get_string(const std::string &key, const std::string &def) const;
  double get_double(const std::string &key, double def) const;
  long get_int(const std::string &key, long def) const;
  bool get_bool(const std::string &key, bool def) const;

  // Required variants throw if the key is missing.
  std::string require_string(const std::string &key) const;

  // Everything that was looked up, with the value actually used. Handy for
  // echoing the effective configuration into logs and snapshot headers.
  const std::map<std::string, std::string> &used() const { return used_; }
  // Keys that were set but never read: usually typos.
  std::vector<std::string> unused() const;

 private:
  std::map<std::string, std::string> values_;
  mutable std::map<std::string, std::string> used_;
};

}  // namespace galactic_tides
