use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign, Index, IndexMut};
use std::f32::consts::PI;
#[cfg(feature = "serde_support")]
use serde::{Serialize, Deserialize};

/// A 2D column vector.
#[derive(Default, Clone, Copy, Debug)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct B2vec2 {
    pub x: f32,
    pub y: f32,
}

impl B2vec2 {
    /// Default constructor does nothing (for performance).
    pub fn new_default() -> B2vec2 {
        return B2vec2::new(0.0, 0.0);
    }

    /// Construct using coordinates.
    pub fn new(x_in: f32, y_in: f32) -> B2vec2 {
        return B2vec2 { x: x_in, y: y_in };
    }

    /// Set this vector to all zeros.
    pub fn set_zero(&mut self) {
        self.x = 0.0;
        self.y = 0.0;
    }

    /// Set this vector to some specified coordinates.
    pub fn set(&mut self, x_: f32, y_: f32) {
        self.x = x_;
        self.y = y_;
    }

    /// Get the length of this vector (the norm).
    pub fn length(self) -> f32 {
        return f32::sqrt(self.x * self.x + self.y * self.y);
    }

    /// Get the length squared. For performance, use this instead of
    /// B2vec2::length (if possible).
    pub fn length_squared(self) -> f32 {
        return self.x * self.x + self.y * self.y;
    }

    /// Convert this vector into a unit vector. Returns the length.
    pub fn normalize(&mut self) -> f32 {
        let length = self.length();
        if length < f32::EPSILON {
            return 0.0;
        }
        let inv_length: f32 = 1.0 / length;
        self.x *= inv_length;
        self.y *= inv_length;

        return length;
    }

    /// Does this vector contain finite coordinates?
    pub fn is_valid(self) -> bool {
        return self.x.is_finite() && self.y.is_finite();
    }

    /// Get the skew vector such that dot(skew_vec, other) == cross(vec, other)
    pub fn skew(self) -> B2vec2 {
        return B2vec2::new(-self.y, self.x);
    }
}

impl Neg for B2vec2 {
    type Output = B2vec2;
    /// Negate this vector.
    fn neg(self) -> B2vec2 {
        return B2vec2 {
            x: -self.x,
            y: -self.y,
        };
    }
}

impl Index<i32> for B2vec2 {
    type Output = f32;

    fn index(&self, index: i32) -> &Self::Output {
        return match index {
            0 => &self.x,
            1 => &self.y,
            _ => { panic!("out bound!") }
        };
    }
}

impl IndexMut<i32> for B2vec2 {
    fn index_mut(&mut self, index: i32) -> &mut Self::Output {
        return match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => { panic!("out bound!") }
        };
    }
}

impl AddAssign<B2vec2> for B2vec2 {
    /// Add a vector to this vector.
    fn add_assign(&mut self, other: B2vec2) {
        self.x += other.x;
        self.y += other.y;
    }
}

impl SubAssign<B2vec2> for B2vec2 {
    /// Subtract a vector from this vector.
    fn sub_assign(&mut self, other: B2vec2) {
        self.x -= other.x;
        self.y -= other.y;
    }
}

impl MulAssign<f32> for B2vec2 {
    /// Multiply this vector by a scalar.
    fn mul_assign(&mut self, other: f32) {
        self.x *= other;
        self.y *= other;
    }
}

//--------------------------------------------------------------------------------------------------
/// A 2D column vector with 3 elements.
#[derive(Default, Clone, Copy, Debug)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct B2Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl B2Vec3 {
    /// Default constructor does nothing (for performance).
    pub fn new_default() -> B2Vec3 {
        return B2Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
    }

    /// Construct using coordinates.
    pub fn new(x_in: f32, y_in: f32, z_in: f32) -> B2Vec3 {
        return B2Vec3 {
            x: x_in,
            y: y_in,
            z: z_in,
        };
    }

    /// Set this vector to all zeros.
    pub fn set_zero(&mut self) {
        self.x = 0.0;
        self.y = 0.0;
        self.z = 0.0;
    }

    /// Set this vector to some specified coordinates.
    pub fn set(&mut self, x_: f32, y_: f32, z_: f32) {
        self.x = x_;
        self.y = y_;
        self.z = z_;
    }
}

impl Neg for B2Vec3 {
    type Output = B2Vec3;
    /// Negate this vector.
    fn neg(self) -> B2Vec3 {
        return B2Vec3::new(-self.x, -self.y, -self.z);
    }
}

impl AddAssign for B2Vec3 {
    /// Add a vector to this vector.
    fn add_assign(&mut self, v: B2Vec3) {
        self.x += v.x;
        self.y += v.y;
        self.z += v.z;
    }
}

impl SubAssign for B2Vec3 {
    /// Subtract a vector from this vector.
    fn sub_assign(&mut self, v: B2Vec3) {
        self.x -= v.x;
        self.y -= v.y;
        self.z -= v.z;
    }
}

impl MulAssign<f32> for B2Vec3 {
    /// Multiply this vector by a scalar.
    fn mul_assign(&mut self, s: f32) {
        self.x *= s;
        self.y *= s;
        self.z *= s;
    }
}

//--------------------------------------------------------------------------------------------------
/// A 2-by-2 matrix. Stored in column-major order.
#[derive(Default, Clone, Copy, Debug)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct B2Mat22 {
    pub ex: B2vec2,
    pub ey: B2vec2,
}

impl B2Mat22 {
    /// The default constructor does nothing (for performance).
    pub fn new_default() -> B2Mat22 {
        return B2Mat22 {
            ex: B2vec2::new_default(),
            ey: B2vec2::new_default(),
        };
    }

    /// Construct this matrix using columns.
    pub fn new(c1: B2vec2, c2: B2vec2) -> B2Mat22 {
        return B2Mat22 { ex: c1, ey: c2 };
    }

    /// Construct this matrix using scalars.
    pub fn new_scalars(a11: f32, a12: f32, a21: f32, a22: f32) -> B2Mat22 {
        return B2Mat22 {
            ex: B2vec2::new(a11, a21),
            ey: B2vec2::new(a12, a22),
        };
    }

    /// initialize this matrix using columns.
    pub fn set(&mut self, c1: B2vec2, c2: B2vec2) {
        self.ex = c1;
        self.ey = c2;
    }

    /// Set this to the identity matrix.
    pub fn set_identity(&mut self) {
        self.ex.x = 1.0;
        self.ey.x = 0.0;
        self.ex.y = 0.0;
        self.ey.y = 1.0;
    }

    /// Set this matrix to all zeros.
    pub fn set_zero(&mut self) {
        self.ex.x = 0.0;
        self.ey.x = 0.0;
        self.ex.y = 0.0;
        self.ey.y = 0.0;
    }

    pub fn get_inverse(&mut self) -> B2Mat22 {
        let a = self.ex.x;
        let b = self.ey.x;
        let c = self.ex.y;
        let d = self.ey.y;
        let mut det: f32 = a * d - b * c;
        if det != 0.0 {
            det = 1.0 / det;
        }
        return B2Mat22 {
            ex: B2vec2::new(det * d, -det * c),
            ey: B2vec2::new(-det * b, det * a),
        };
    }

    /// solve A * x = b, where b is a column vector. This is more efficient
    /// than computing the inverse in one-shot cases.
    pub fn solve(self, b: B2vec2) -> B2vec2 {
        let a11 = self.ex.x;
        let a12 = self.ey.x;
        let a21 = self.ex.y;
        let a22 = self.ey.y;
        let mut det: f32 = a11 * a22 - a12 * a21;
        if det != 0.0 {
            det = 1.0 / det;
        }
        let x: B2vec2;
        x = B2vec2 {
            x: det * (a22 * b.x - a12 * b.y),
            y: det * (a11 * b.y - a21 * b.x),
        };
        return x;
    }
}

//--------------------------------------------------------------------------------------------------
/// A 3-by-3 matrix. Stored in column-major order.
#[derive(Default, Clone, Copy, Debug)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct B2Mat33 {
    pub ex: B2Vec3,
    pub ey: B2Vec3,
    pub ez: B2Vec3,
}

impl B2Mat33 {
    /// The default constructor does nothing (for performance).
    pub fn new_default() -> B2Mat33 {
        return B2Mat33 {
            ex: B2Vec3::new_default(),
            ey: B2Vec3::new_default(),
            ez: B2Vec3::new_default(),
        };
    }

    /// Construct this matrix using columns.
    pub fn new(c1: B2Vec3, c2: B2Vec3, c3: B2Vec3) -> B2Mat33 {
        return B2Mat33 {
            ex: c1,
            ey: c2,
            ez: c3,
        };
    }

    /// Set this matrix to all zeros.
    pub fn set_zero(&mut self) {
        self.ex.set_zero();
        self.ey.set_zero();
        self.ez.set_zero();
    }

    /// solve A * x = b, where b is a column vector. This is more efficient
    /// than computing the inverse in one-shot cases.
    pub fn solve33(self, b: B2Vec3) -> B2Vec3 {
        let mut det: f32 = b2_dot_vec3(self.ex, b2_cross_vec3(self.ey, self.ez));
        if det != 0.0 {
            det = 1.0 / det;
        }
        let x = B2Vec3 {
            x: det * b2_dot_vec3(b, b2_cross_vec3(self.ey, self.ez)),
            y: det * b2_dot_vec3(self.ex, b2_cross_vec3(b, self.ez)),
            z: det * b2_dot_vec3(self.ex, b2_cross_vec3(self.ey, b)),
        };
        return x;
    }

    /// solve A * x = b, where b is a column vector. This is more efficient
    /// than computing the inverse in one-shot cases. solve only the upper
    /// 2-by-2 matrix equation.
    pub fn solve22(self, b: B2vec2) -> B2vec2 {
        let (a11, a12, a21, a22) = (self.ex.x, self.ey.x, self.ex.y, self.ey.y);
        let mut det: f32 = a11 * a22 - a12 * a21;
        if det != 0.0 {
            det = 1.0 / det;
        }
        let x = B2vec2 {
            x: det * (a22 * b.x - a12 * b.y),
            y: det * (a11 * b.y - a21 * b.x),
        };
        return x;
    }

    /// Get the inverse of this matrix as a 2-by-2.
    /// Returns the zero matrix if singular.
    pub fn get_inverse22(self, m: &mut B2Mat33) {
        let (a, b, c, d) = (self.ex.x, self.ey.x, self.ex.y, self.ey.y);
        let mut det: f32 = a * d - b * c;
        if det != 0.0 {
            det = 1.0 / det;
        }

        m.ex.x = det * d;
        m.ey.x = -det * b;
        m.ex.z = 0.0;
        m.ex.y = -det * c;
        m.ey.y = det * a;
        m.ey.z = 0.0;
        m.ez.x = 0.0;
        m.ez.y = 0.0;
        m.ez.z = 0.0;
    }

    /// Get the symmetric inverse of this matrix as a 3-by-3.
    /// Returns the zero matrix if singular.
    pub fn get_sym_inverse33(self, m: &mut B2Mat33) {
        let mut det = b2_dot_vec3(self.ex, b2_cross_vec3(self.ey, self.ez));
        if det != 0.0 {
            det = 1.0 / det;
        }

        let (a11, a12, a13) = (self.ex.x, self.ey.x, self.ez.x);
        let (a22, a23) = (self.ey.y, self.ez.y);
        let a33 = self.ez.z;

        m.ex.x = det * (a22 * a33 - a23 * a23);
        m.ex.y = det * (a13 * a23 - a12 * a33);
        m.ex.z = det * (a12 * a23 - a13 * a22);

        m.ey.x = m.ex.y;
        m.ey.y = det * (a11 * a33 - a13 * a13);
        m.ey.z = det * (a13 * a12 - a11 * a23);

        m.ez.x = m.ex.z;
        m.ez.y = m.ey.z;
        m.ez.z = det * (a11 * a22 - a12 * a12);
    }
}

//--------------------------------------------------------------------------------------------------
/// Rotation
#[derive(Clone, Default, Copy, Debug)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct B2Rot {
    /// Sine and cosine
    s: f32,
    c: f32,
}

impl B2Rot {
    pub fn new_default() -> B2Rot {
        return B2Rot {
            s: 0.0,
            c: 0.0,
        };
    }

    /// initialize from an angle in radians
    pub fn new(angle: f32) -> B2Rot {
        // TODO_ERIN optimize
        return B2Rot {
            s: f32::sin(angle),
            c: f32::cos(angle),
        };
    }

    /// Set using an angle in radians.
    pub fn set(&mut self, angle: f32) {
        // TODO_ERIN optimize
        self.s = f32::sin(angle);
        self.c = f32::cos(angle);
    }

    /// Set to the identity rotation
    pub fn set_identity(&mut self) {
        self.s = 0.0;
        self.c = 1.0;
    }

    /// Get the angle in radians
    pub fn get_angle(self) -> f32 {
        return f32::atan2(self.s, self.c);
    }

    /// Get the x-axis
    pub fn get_xaxis(self) -> B2vec2 {
        return B2vec2::new(self.c, self.s);
    }

    /// Get the u-axis
    pub fn get_yaxis(self) -> B2vec2 {
        return B2vec2::new(-self.s, self.c);
    }
}

//--------------------------------------------------------------------------------------------------
/// A transform contains translation and rotation. It is used to represent
/// the position and orientation of rigid frames.
#[derive(Default, Clone, Copy, Debug)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct B2Transform {
    pub p: B2vec2,
    pub q: B2Rot,
}

impl B2Transform {
    /// The default constructor does nothing.
    pub fn new_default() -> B2Transform {
        return B2Transform {
            p: B2vec2::new_default(),
            q: B2Rot::new_default(),
        };
    }

    /// initialize using a position vector and a rotation.
    pub fn new(position: B2vec2, rotation: B2Rot) -> B2Transform {
        return B2Transform {
            p: position,
            q: rotation,
        };
    }

    /// Set this to the identity transform.
    pub fn set_identity(&mut self) {
        self.p.set_zero();
        self.q.set_identity();
    }

    /// Set this based on the position and angle.
    pub fn set(&mut self, position: B2vec2, angle: f32) {
        self.p = position;
        self.q.set(angle);
    }
}

//--------------------------------------------------------------------------------------------------
/// This describes the motion of a body/shape for TOI computation.
/// Shapes are defined with respect to the body origin, which may
/// no coincide with the center of mass. However, to support dynamics
/// we must interpolate the center of mass position.
#[derive(Default, Clone, Copy, Debug)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct B2Sweep {
    //< local center of mass position
    pub local_center: B2vec2,
    //< center world positions
    pub c0: B2vec2,
    pub c: B2vec2,
    //< world angles
    pub a0: f32,
    pub a: f32,

    // Fraction of the current time step in the range [0,1]
    // c0 and a0 are the positions at alpha0.
    pub alpha0: f32,
}

impl B2Sweep {
    /// Get the interpolated transform at a specific time.
    /// * `transform` - the output transform
    /// * `beta` - is a factor in [0,1], where 0 indicates alpha0.
    pub fn get_transform(self, transform: &mut B2Transform, beta: f32) {
        transform.p = self.c0 + beta * (self.c - self.c0);
        let angle: f32 = self.a0 + beta * (self.a - self.a0);
        transform.q.set(angle);

        // Shift to origin
        transform.p -= b2_mul_rot_by_vec2(transform.q, self.local_center);
    }

    /// advance the sweep forward, yielding a new initial state.
    /// * `alpha` - the new initial time.
    pub fn advance(&mut self, alpha: f32) {
        debug_assert!(self.alpha0 < 1.0);
        let beta: f32 = (alpha - self.alpha0) / (1.0 - self.alpha0);
        self.c0 += beta * (self.c - self.c0);
        self.a0 += beta * (self.a - self.a0);
        self.alpha0 = alpha;
    }

    /// normalize the angles.
    pub fn normalize(&mut self) {
        let two_pi: f32 = 2.0 * PI;
        let d: f32 = two_pi * f32::floor(self.a0 / two_pi);
        self.a0 -= d;
        self.a -= d;
    }
}

//--------------------------------------------------------------------------------------------------
/// Perform the dot product on two vectors.
pub fn b2_dot(a: B2vec2, b: B2vec2) -> f32 {
    return a.x * b.x + a.y * b.y;
}

/// Perform the cross product on two vectors. In 2D this produces a scalar.
pub fn b2_cross(a: B2vec2, b: B2vec2) -> f32 {
    return a.x * b.y - a.y * b.x;
}

/// Perform the cross product on a vector and a scalar. In 2D this produces
/// a vector.
pub fn b2_cross_vec_by_scalar(a: B2vec2, s: f32) -> B2vec2 {
    return B2vec2::new(s * a.y, -s * a.x);
}

/// Perform the cross product on a scalar and a vector. In 2D this produces
/// a vector.
pub fn b2_cross_scalar_by_vec(s: f32, a: B2vec2) -> B2vec2 {
    return B2vec2::new(-s * a.y, s * a.x);
}

/// Multiply a matrix times a vector. If a rotation matrix is provided,
/// then this transforms the vector from one frame to another.
pub fn b2_mul(a: B2Mat22, v: B2vec2) -> B2vec2 {
    return B2vec2::new(a.ex.x * v.x + a.ey.x * v.y, a.ex.y * v.x + a.ey.y * v.y);
}

/// Multiply a matrix transpose times a vector. If a rotation matrix is provided,
/// then this transforms the vector from one frame to another (inverse transform).
pub fn b2_mul_t(a: B2Mat22, v: B2vec2) -> B2vec2 {
    return B2vec2::new(b2_dot(v, a.ex), b2_dot(v, a.ey));
}

impl Add for B2vec2 {
    type Output = B2vec2;
    /// Add two vectors component-wise.
    fn add(self, b: B2vec2) -> B2vec2 {
        return B2vec2::new(self.x + b.x, self.y + b.y);
    }
}

impl Sub for B2vec2 {
    type Output = B2vec2;
    /// Subtract two vectors component-wise.
    fn sub(self, b: B2vec2) -> B2vec2 {
        return B2vec2::new(self.x - b.x, self.y - b.y);
    }
}

impl Mul<B2vec2> for f32 {
    type Output = B2vec2;
    fn mul(self, a: B2vec2) -> B2vec2 {
        return B2vec2::new(self * a.x, self * a.y);
    }
}

impl PartialEq for B2vec2 {
    fn eq(&self, other: &Self) -> bool {
        return self.x == other.x && self.y == other.y;
    }
}

impl Eq for B2vec2 {}

pub fn b2_distance_vec2(a: B2vec2, b: B2vec2) -> f32 {
    let c: B2vec2 = a - b;
    return c.length();
}

pub fn b2_distance_vec2_squared(a: B2vec2, b: B2vec2) -> f32 {
    let c: B2vec2 = a - b;
    return b2_dot(c, c);
}

impl Mul<B2Vec3> for f32 {
    type Output = B2Vec3;

    fn mul(self, a: B2Vec3) -> B2Vec3 {
        return B2Vec3::new(self * a.x, self * a.y, self * a.z);
    }
}

impl Add for B2Vec3 {
    type Output = B2Vec3;
    /// Add two vectors component-wise.
    fn add(self, b: B2Vec3) -> B2Vec3 {
        return B2Vec3::new(self.x + b.x, self.y + b.y, self.z + b.z);
    }
}

/// Subtract two vectors component-wise.
impl Sub for B2Vec3 {
    type Output = B2Vec3;
    /// Add two vectors component-wise.
    fn sub(self, b: B2Vec3) -> B2Vec3 {
        return B2Vec3::new(self.x - b.x, self.y - b.y, self.z - b.z);
    }
}

/// Perform the dot product on two vectors.
pub fn b2_dot_vec3(a: B2Vec3, b: B2Vec3) -> f32 {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/// Perform the cross product on two vectors.
pub fn b2_cross_vec3(a: B2Vec3, b: B2Vec3) -> B2Vec3 {
    return B2Vec3::new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    );
}

impl Add for B2Mat22 {
    type Output = B2Mat22;
    fn add(self, b: B2Mat22) -> B2Mat22 {
        return B2Mat22::new(self.ex + b.ex, self.ey + b.ey);
    }
}

// A * b
pub fn b2_mul_mat22(a: B2Mat22, b: B2Mat22) -> B2Mat22 {
    return B2Mat22::new(b2_mul(a, b.ex), b2_mul(a, b.ey));
}

// A^T * b
pub fn b2_mul_t_mat22(a: B2Mat22, b: B2Mat22) -> B2Mat22 {
    let c1 = B2vec2::new(b2_dot(a.ex, b.ex), b2_dot(a.ey, b.ex));
    let c2 = B2vec2::new(b2_dot(a.ex, b.ey), b2_dot(a.ey, b.ey));
    return B2Mat22::new(c1, c2);
}

/// Multiply a matrix times a vector.
pub fn b2_mul_mat33(a: B2Mat33, v: B2Vec3) -> B2Vec3 {
    return v.x * a.ex + v.y * a.ey + v.z * a.ez;
}

/// Multiply a matrix times a vector.
pub fn b2_mul22(a: B2Mat33, v: B2vec2) -> B2vec2 {
    return B2vec2::new(a.ex.x * v.x + a.ey.x * v.y, a.ex.y * v.x + a.ey.y * v.y);
}

/// Multiply two rotations: q * r
pub fn b2_mul_rot(q: B2Rot, r: B2Rot) -> B2Rot {
    // [qc -qs] * [rc -rs] = [qc*rc-qs*rs -qc*rs-qs*rc]
    // [qs  qc]   [rs  rc]   [qs*rc+qc*rs -qs*rs+qc*rc]
    // s = qs * rc + qc * rs
    // c = qc * rc - qs * rs
    let qr = B2Rot {
        s: q.s * r.c + q.c * r.s,
        c: q.c * r.c - q.s * r.s,
    };
    return qr;
}

/// Transpose multiply two rotations: qT * r
pub fn b2_mul_t_rot(q: B2Rot, r: B2Rot) -> B2Rot {
    // [ qc qs] * [rc -rs] = [qc*rc+qs*rs -qc*rs+qs*rc]
    // [-qs qc]   [rs  rc]   [-qs*rc+qc*rs qs*rs+qc*rc]
    // s = qc * rs - qs * rc
    // c = qc * rc + qs * rs
    let qr = B2Rot {
        s: q.c * r.s - q.s * r.c,
        c: q.c * r.c + q.s * r.s,
    };
    return qr;
}

/// Rotate a vector
pub fn b2_mul_rot_by_vec2(q: B2Rot, v: B2vec2) -> B2vec2 {
    return B2vec2::new(q.c * v.x - q.s * v.y, q.s * v.x + q.c * v.y);
}

/// Inverse rotate a vector
pub fn b2_mul_t_rot_by_vec2(q: B2Rot, v: B2vec2) -> B2vec2 {
    return B2vec2::new(q.c * v.x + q.s * v.y, -q.s * v.x + q.c * v.y);
}

pub fn b2_mul_transform_by_vec2(t: B2Transform, v: B2vec2) -> B2vec2 {
    let x: f32 = (t.q.c * v.x - t.q.s * v.y) + t.p.x;
    let y: f32 = (t.q.s * v.x + t.q.c * v.y) + t.p.y;

    return B2vec2::new(x, y);
}

pub fn b2_mul_t_transform_by_vec2(t: B2Transform, v: B2vec2) -> B2vec2 {
    let px: f32 = v.x - t.p.x;
    let py: f32 = v.y - t.p.y;
    let x: f32 = t.q.c * px + t.q.s * py;
    let y: f32 = -t.q.s * px + t.q.c * py;

    return B2vec2::new(x, y);
}

// v2 = A.q.Rot(b.q.Rot(v1) + b.p) + A.p
//    = (A.q * b.q).Rot(v1) + A.q.Rot(b.p) + A.p
pub fn b2_mul_transform(a: B2Transform, b: B2Transform) -> B2Transform {
    let c = B2Transform {
        q: b2_mul_rot(a.q.clone(), b.q.clone()),
        p: b2_mul_rot_by_vec2(a.q, b.p) + a.p,
    };
    return c;
}

// v2 = A.q' * (b.q * v1 + b.p - A.p)
//    = A.q' * b.q * v1 + A.q' * (b.p - A.p)
pub fn b2_mul_t_transform(a: B2Transform, b: B2Transform) -> B2Transform {
    let c = B2Transform {
        q: b2_mul_t_rot(a.q, b.q),
        p: b2_mul_t_rot_by_vec2(a.q, b.p - a.p),
    };
    return c;
}

pub fn b2_abs_vec2(a: B2vec2) -> B2vec2 {
    return B2vec2::new(f32::abs(a.x), f32::abs(a.y));
}

pub fn b2_abs_mat22(a: B2Mat22) -> B2Mat22 {
    return B2Mat22::new(b2_abs_vec2(a.ex), b2_abs_vec2(a.ey));
}

pub fn b2_min_vec2(a: B2vec2, b: B2vec2) -> B2vec2 {
    return B2vec2::new(f32::min(a.x, b.x), f32::min(a.y, b.y));
}

pub fn b2_max_vec2(a: B2vec2, b: B2vec2) -> B2vec2 {
    return B2vec2::new(f32::max(a.x, b.x), f32::max(a.y, b.y));
}

pub fn b2_clamp_vec2(a: B2vec2, low: B2vec2, high: B2vec2) -> B2vec2 {
    return b2_max_vec2(low, b2_max_vec2(a, high));
}

/// "Next Largest Power of 2
/// Given a binary integer value x, the next largest power of 2 can be computed by a SWAR algorithm
/// that recursively "folds" the upper bits into the lower bits. This process yields a bit vector with
/// the same most significant 1 as x, but all 1's below it. Adding 1 to that value yields the next
/// largest power of 2. For a 32-bit value:"
pub fn b2_next_power_of_two(v: u32) -> u32 {
    let mut x: u32 = v;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

pub fn b2_is_power_of_two(x: u32) -> bool {
    let result: bool = x > 0 && (x & (x - 1)) == 0;
    return result;
}
