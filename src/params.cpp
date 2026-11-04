#include "galactic_tides/params.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace galactic_tides {

namespace {

std::string trim(const std::string &s) {
  auto b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return {};
  auto e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

}  // namespace

void Params::load_file(const std::string &path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open parameter file: " + path);
  std::stringstream ss;
  ss << in.rdbuf();
  load_string(ss.str(), path);
}

void Params::load_string(const std::string &text, const std::string &origin) {
  std::istringstream in(text);
  std::string line;
  int lineno = 0;
  while (std::getline(in, line)) {
    ++lineno;
    auto hash = line.find('#');
    if (hash != std::string::npos) line.erase(hash);
    line = trim(line);
    if (line.empty()) continue;
    auto eq = line.find('=');
    if (eq == std::string::npos) {
      throw std::runtime_error(origin + ":" + std::to_string(lineno) +
                               ": expected 'key = value', got '" + line + "'");
    }
    values_[trim(line.substr(0, eq))] = trim(line.substr(eq + 1));
  }
}

std::vector<std::string> Params::apply_overrides(const std::vector<std::string> &tokens) {
  std::vector<std::string> rest;
  for (const auto &t : tokens) {
    auto eq = t.find('=');
    if (eq == std::string::npos || eq == 0) {
      rest.push_back(t);
      continue;
    }
    values_[trim(t.substr(0, eq))] = trim(t.substr(eq + 1));
  }
  return rest;
}

std::string Params::get_string(const std::string &key, const std::string &def) const {
  auto it = values_.find(key);
  std::string v = it == values_.end() ? def : it->second;
  used_[key] = v;
  return v;
}

double Params::get_double(const std::string &key, double def) const {
  auto it = values_.find(key);
  if (it == values_.end()) {
    used_[key] = std::to_string(def);
    return def;
  }
  try {
    std::size_t pos = 0;
    double v = std::stod(it->second, &pos);
    if (pos != it->second.size()) throw std::invalid_argument("trailing");
    used_[key] = it->second;
    return v;
  } catch (const std::exception &) {
    throw std::runtime_error("parameter '" + key + "': not a number: '" + it->second + "'");
  }
}

long Params::get_int(const std::string &key, long def) const {
  auto it = values_.find(key);
  if (it == values_.end()) {
    used_[key] = std::to_string(def);
    return def;
  }
  try {
    std::size_t pos = 0;
    long v = std::stol(it->second, &pos);
    if (pos != it->second.size()) throw std::invalid_argument("trailing");
    used_[key] = it->second;
    return v;
  } catch (const std::exception &) {
    throw std::runtime_error("parameter '" + key + "': not an integer: '" + it->second + "'");
  }
}

bool Params::get_bool(const std::string &key, bool def) const {
  auto it = values_.find(key);
  if (it == values_.end()) {
    used_[key] = def ? "true" : "false";
    return def;
  }
  std::string v = it->second;
  std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return std::tolower(c); });
  used_[key] = it->second;
  if (v == "1" || v == "true" || v == "yes" || v == "on") return true;
  if (v == "0" || v == "false" || v == "no" || v == "off") return false;
  throw std::runtime_error("parameter '" + key + "': not a boolean: '" + it->second + "'");
}

std::string Params::require_string(const std::string &key) const {
  auto it = values_.find(key);
  if (it == values_.end()) throw std::runtime_error("missing required parameter: " + key);
  used_[key] = it->second;
  return it->second;
}

std::vector<std::string> Params::unused() const {
  std::vector<std::string> out;
  for (const auto &kv : values_)
    if (!used_.count(kv.first)) out.push_back(kv.first);
  return out;
}

}  // namespace galactic_tides
