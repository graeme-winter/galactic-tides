#pragma once

#include <cmath>

namespace galactic_tides {

struct Vec3 {
  double x = 0.0, y = 0.0, z = 0.0;

  constexpr Vec3() = default;
  constexpr Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

  constexpr double &operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }
  constexpr double operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }

  constexpr Vec3 &operator+=(const Vec3 &o) { x += o.x; y += o.y; z += o.z; return *this; }
  constexpr Vec3 &operator-=(const Vec3 &o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
  constexpr Vec3 &operator*=(double s) { x *= s; y *= s; z *= s; return *this; }
  constexpr Vec3 &operator/=(double s) { x /= s; y /= s; z /= s; return *this; }
};

constexpr Vec3 operator+(Vec3 a, const Vec3 &b) { return a += b; }
constexpr Vec3 operator-(Vec3 a, const Vec3 &b) { return a -= b; }
constexpr Vec3 operator-(const Vec3 &a) { return {-a.x, -a.y, -a.z}; }
constexpr Vec3 operator*(Vec3 a, double s) { return a *= s; }
constexpr Vec3 operator*(double s, Vec3 a) { return a *= s; }
constexpr Vec3 operator/(Vec3 a, double s) { return a /= s; }

constexpr double dot(const Vec3 &a, const Vec3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
constexpr Vec3 cross(const Vec3 &a, const Vec3 &b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
constexpr double norm2(const Vec3 &a) { return dot(a, a); }
inline double norm(const Vec3 &a) { return std::sqrt(norm2(a)); }

}  // namespace galactic_tides
