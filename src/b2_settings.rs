#[cfg(feature = "serde_support")]
use serde::{de::DeserializeOwned, Serialize};
use std::f32::consts::PI;

// Tunable Constants

/// You can use this to change the length scale used by your game.
/// For example for inches you could use 39.4.
pub const B2_LENGTH_UNITS_PER_METER: f32 = 1.0;

/// The maximum number of vertices on a convex polygon. You cannot increase
/// this too much because b2BlockAllocator has a maximum object size.
pub const B2_MAX_POLYGON_VERTICES: usize = 8;

// User data
/// You can define this to inject whatever data you want in b2Body, b2Fixture, b2Joint
#[cfg(not(feature = "serde_support"))]
pub trait UserDataType: Default + Clone + 'static {
    type Fixture: Default + Clone + std::fmt::Debug;
    type Body: Default + Clone + PartialEq + std::fmt::Debug;
    type Joint: Default + Clone + std::fmt::Debug;
}

#[cfg(feature = "serde_support")]
pub trait UserDataType: Default + Clone + Serialize + DeserializeOwned + 'static {
    type Fixture: Default + Clone + Serialize + DeserializeOwned + std::fmt::Debug;
    type Body: Default + Clone + Serialize + DeserializeOwned + PartialEq + std::fmt::Debug;
    type Joint: Default + Clone + Serialize + DeserializeOwned + std::fmt::Debug;
}

//--------------------------------------------------------------------------------------------------
/// Global tuning constants based on meters-kilograms-seconds (MKS) units.

// Collision

/// The maximum number of contact points between two convex shapes. Do
/// not change this value.
pub const B2_MAX_MANIFOLD_POINTS: usize = 2;

/// This is used to fatten AABBs in the dynamic tree. This allows proxies
/// to move by a small amount without triggering a tree adjustment.
/// This is in meters.
pub const B2_AABB_EXTENSION: f32 = 0.1 * B2_LENGTH_UNITS_PER_METER;

/// This is used to fatten AABBs in the dynamic tree. This is used to predict
/// the future position based on the current displacement.
/// This is a dimensionless multiplier.
pub const B2_AABB_MULTIPLIER: f32 = 4.0;

/// A small length used as a collision and constraint tolerance. Usually it is
/// chosen to be numerically significant, but visually insignificant.
pub const B2_LINEAR_SLOP: f32 = 0.005 * B2_LENGTH_UNITS_PER_METER;

/// A small angle used as a collision and constraint tolerance. Usually it is
/// chosen to be numerically significant, but visually insignificant.
pub const B2_ANGULAR_SLOP: f32 = 2.0 / 180.0 * PI;

/// The radius of the polygon/edge shape skin. This should not be modified. Making
/// this smaller means polygons will have an insufficient buffer for continuous collision.
/// Making it larger may create artifacts for vertex collision.
pub const B2_POLYGON_RADIUS: f32 = 2.0 * B2_LINEAR_SLOP;

/// Maximum number of sub-steps per contact in continuous physics simulation.
pub const B2_MAX_SUB_STEPS: usize = 8;

// Dynamics

/// Maximum number of contacts to be handled to solve a TOI impact.
pub const B2_MAX_TOICONTACTS: usize = 32;

/// The maximum linear position correction used when solving constraints. This helps to
/// prevent overshoot.
pub const B2_MAX_LINEAR_CORRECTION: f32 = 0.2 * B2_LENGTH_UNITS_PER_METER;

/// The maximum angular position correction used when solving constraints. This helps to
/// prevent overshoot.
pub const B2_MAX_ANGULAR_CORRECTION: f32 = 8.0 / 180.0 * PI;

/// The maximum linear velocity of a body. This limit is very large and is used
/// to prevent numerical problems. You shouldn't need to adjust this.
pub const B2_MAX_TRANSLATION: f32 = 2.0 * B2_LENGTH_UNITS_PER_METER;
pub const B2_MAX_TRANSLATION_SQUARED: f32 = B2_MAX_TRANSLATION * B2_MAX_TRANSLATION;

/// The maximum angular velocity of a body. This limit is very large and is used
/// to prevent numerical problems. You shouldn't need to adjust this.
pub const B2_MAX_ROTATION: f32 = 0.5 * PI;
pub const B2_MAX_ROTATION_SQUARED: f32 = B2_MAX_ROTATION * B2_MAX_ROTATION;

/// This scale factor controls how fast overlap is resolved. Ideally this would be 1 so
/// that overlap is removed in one time step. However using values close to 1 often lead
/// to overshoot.
pub const B2_BAUMGARTE: f32 = 0.2;
pub const B2_TOI_BAUMGARTE: f32 = 0.75;

// Sleep

/// The time that a body must be still before it will go to sleep.
pub const B2_TIME_TO_SLEEP: f32 = 0.5;

/// A body cannot sleep if its linear velocity is above this tolerance.
pub const B2_LINEAR_SLEEP_TOLERANCE: f32 = 0.01 * B2_LENGTH_UNITS_PER_METER;

/// A body cannot sleep if its angular velocity is above this tolerance.
pub const B2_ANGULAR_SLEEP_TOLERANCE: f32 = 2.0 / 180.0 * PI;
