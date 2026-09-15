//! Device scheduling and memory-aware admission control (PRD §18.3).
//!
//! The distributed-execution section requires that "impossible configurations
//! are rejected before scheduling, not after a worker OOMs," that admission
//! control "tracks per-device outstanding allocation and refuses to lease a
//! trial whose declared `ResourceSpec` would exceed a device's remaining
//! budget," and that GPU assignment goes through a `DeviceScheduler` handing out
//! exclusive or shared (MPS-style) device handles.
//!
//! This module provides that mechanism as pure resource accounting over
//! [`ResourceSpec`], independent of any real hardware, so it can be unit-tested
//! and wired into the distributed executor when it lands (v0.6). A
//! [`DeviceLease`] reserves capacity on acquisition and releases it on drop
//! (RAII), and fractional GPU/CPU requests model shared devices directly.

use crate::executor::ResourceSpec;
use std::sync::{Arc, Mutex};

/// Why a trial could not be admitted to any device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionError {
    /// No single device has enough *total* capacity to ever satisfy the request
    /// — an impossible configuration that should be rejected outright (§18.3).
    Impossible,
    /// Every device that could fit the request is currently too busy; retry once
    /// outstanding leases are released.
    NoCapacityAvailable,
}

impl std::fmt::Display for AdmissionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AdmissionError::Impossible => {
                write!(f, "no device can ever satisfy this resource request")
            }
            AdmissionError::NoCapacityAvailable => {
                write!(f, "no device currently has capacity for this request")
            }
        }
    }
}

impl std::error::Error for AdmissionError {}

/// A device with a fixed total capacity.
#[derive(Debug, Clone)]
pub struct Device {
    /// Stable device identifier (e.g. "gpu:0").
    pub id: String,
    /// The device's total capacity.
    pub capacity: ResourceSpec,
}

impl Device {
    /// A device with the given id and capacity.
    pub fn new(id: impl Into<String>, capacity: ResourceSpec) -> Self {
        Device {
            id: id.into(),
            capacity,
        }
    }
}

/// Internal per-device state: total capacity and currently reserved usage.
#[derive(Debug)]
struct DeviceState {
    id: String,
    capacity: ResourceSpec,
    used: ResourceSpec,
}

impl DeviceState {
    /// Whether `req` fits in the remaining (capacity - used) budget.
    fn fits(&self, req: &ResourceSpec) -> bool {
        le(&add(&self.used, req), &self.capacity)
    }

    /// Whether `req` could ever fit in this device's total capacity.
    fn could_ever_fit(&self, req: &ResourceSpec) -> bool {
        le(req, &self.capacity)
    }
}

/// Hands out exclusive or shared device leases with memory-aware admission
/// control (§18.3). Cloneable and thread-safe; clones share the same accounting.
#[derive(Clone)]
pub struct DeviceScheduler {
    inner: Arc<Mutex<Vec<DeviceState>>>,
}

impl DeviceScheduler {
    /// A scheduler over the given devices.
    pub fn new(devices: impl IntoIterator<Item = Device>) -> Self {
        let states = devices
            .into_iter()
            .map(|d| DeviceState {
                id: d.id,
                capacity: d.capacity,
                used: zero(),
            })
            .collect();
        DeviceScheduler {
            inner: Arc::new(Mutex::new(states)),
        }
    }

    /// Whether any device could ever satisfy `req` (ignoring current load).
    pub fn is_feasible(&self, req: &ResourceSpec) -> bool {
        self.inner
            .lock()
            .unwrap()
            .iter()
            .any(|d| d.could_ever_fit(req))
    }

    /// Try to reserve capacity for `req`, returning a [`DeviceLease`] that
    /// releases it on drop. Distinguishes an impossible request from a transient
    /// lack of capacity (§18.3).
    ///
    /// Best-fit: among devices that currently fit the request, the one with the
    /// least remaining CPU is chosen, packing trials rather than fragmenting.
    pub fn try_admit(&self, req: &ResourceSpec) -> Result<DeviceLease, AdmissionError> {
        let mut devices = self.inner.lock().unwrap();
        if !devices.iter().any(|d| d.could_ever_fit(req)) {
            return Err(AdmissionError::Impossible);
        }
        // Best-fit by smallest remaining CPU headroom among devices that fit.
        let choice = devices
            .iter()
            .enumerate()
            .filter(|(_, d)| d.fits(req))
            .min_by(|(_, a), (_, b)| {
                let ra = a.capacity.cpus - a.used.cpus;
                let rb = b.capacity.cpus - b.used.cpus;
                ra.partial_cmp(&rb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);

        match choice {
            Some(i) => {
                devices[i].used = add(&devices[i].used, req);
                let id = devices[i].id.clone();
                Ok(DeviceLease {
                    scheduler: self.inner.clone(),
                    device_id: id,
                    reserved: req.clone(),
                    active: true,
                })
            }
            None => Err(AdmissionError::NoCapacityAvailable),
        }
    }

    /// Currently reserved usage on a device by id, for inspection/tests.
    pub fn used(&self, device_id: &str) -> Option<ResourceSpec> {
        self.inner
            .lock()
            .unwrap()
            .iter()
            .find(|d| d.id == device_id)
            .map(|d| d.used.clone())
    }
}

/// A reservation of capacity on a specific device. Releases the reserved
/// resources back to the scheduler when dropped.
pub struct DeviceLease {
    scheduler: Arc<Mutex<Vec<DeviceState>>>,
    device_id: String,
    reserved: ResourceSpec,
    active: bool,
}

impl DeviceLease {
    /// The device this lease is bound to.
    pub fn device_id(&self) -> &str {
        &self.device_id
    }

    /// The reserved resources.
    pub fn reserved(&self) -> &ResourceSpec {
        &self.reserved
    }

    /// Release the lease early (also happens automatically on drop).
    pub fn release(mut self) {
        self.release_inner();
    }

    fn release_inner(&mut self) {
        if !self.active {
            return;
        }
        self.active = false;
        if let Ok(mut devices) = self.scheduler.lock() {
            if let Some(d) = devices.iter_mut().find(|d| d.id == self.device_id) {
                d.used = sub(&d.used, &self.reserved);
            }
        }
    }
}

impl Drop for DeviceLease {
    fn drop(&mut self) {
        self.release_inner();
    }
}

// ----- ResourceSpec arithmetic (componentwise over cpus/gpus/memory/custom) ---

/// A zero resource spec — the correct starting point for *usage* accounting.
/// (`ResourceSpec::default()` is a default *request* of one CPU, not zero.)
fn zero() -> ResourceSpec {
    ResourceSpec {
        cpus: 0.0,
        gpus: 0.0,
        memory_mb: 0,
        custom: std::collections::BTreeMap::new(),
    }
}

fn add(a: &ResourceSpec, b: &ResourceSpec) -> ResourceSpec {
    let mut custom = a.custom.clone();
    for (k, v) in &b.custom {
        *custom.entry(k.clone()).or_insert(0.0) += v;
    }
    ResourceSpec {
        cpus: a.cpus + b.cpus,
        gpus: a.gpus + b.gpus,
        memory_mb: a.memory_mb + b.memory_mb,
        custom,
    }
}

fn sub(a: &ResourceSpec, b: &ResourceSpec) -> ResourceSpec {
    let mut custom = a.custom.clone();
    for (k, v) in &b.custom {
        let e = custom.entry(k.clone()).or_insert(0.0);
        *e = (*e - v).max(0.0);
    }
    ResourceSpec {
        cpus: (a.cpus - b.cpus).max(0.0),
        gpus: (a.gpus - b.gpus).max(0.0),
        memory_mb: a.memory_mb.saturating_sub(b.memory_mb),
        custom,
    }
}

/// Whether `a <= b` componentwise (custom keys in `a` must not exceed `b`'s,
/// treating a key absent from `b` as zero). A small epsilon tolerates float
/// rounding on the CPU/GPU axes.
fn le(a: &ResourceSpec, b: &ResourceSpec) -> bool {
    const EPS: f64 = 1e-9;
    if a.cpus > b.cpus + EPS || a.gpus > b.gpus + EPS || a.memory_mb > b.memory_mb {
        return false;
    }
    a.custom
        .iter()
        .all(|(k, v)| *v <= b.custom.get(k).copied().unwrap_or(0.0) + EPS)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scheduler() -> DeviceScheduler {
        DeviceScheduler::new([
            Device::new(
                "gpu:0",
                ResourceSpec::cpus(4.0).with_gpus(1.0).with_memory_mb(8000),
            ),
            Device::new(
                "gpu:1",
                ResourceSpec::cpus(4.0).with_gpus(1.0).with_memory_mb(8000),
            ),
        ])
    }

    #[test]
    fn admits_until_capacity_then_refuses() {
        let s = scheduler();
        let req = ResourceSpec::cpus(2.0).with_gpus(0.5).with_memory_mb(4000);
        // Two devices, each fits two of these -> four leases total.
        let l1 = s.try_admit(&req).unwrap();
        let l2 = s.try_admit(&req).unwrap();
        let l3 = s.try_admit(&req).unwrap();
        let l4 = s.try_admit(&req).unwrap();
        // Fifth exceeds all remaining capacity.
        assert!(matches!(
            s.try_admit(&req),
            Err(AdmissionError::NoCapacityAvailable)
        ));

        // Releasing one frees room for exactly one more.
        drop(l1);
        let l5 = s.try_admit(&req).unwrap();
        assert!(matches!(
            s.try_admit(&req),
            Err(AdmissionError::NoCapacityAvailable)
        ));
        drop((l2, l3, l4, l5));
    }

    #[test]
    fn rejects_impossible_configuration() {
        let s = scheduler();
        // No device has 32 GB, so this can never be scheduled.
        let too_big = ResourceSpec::cpus(1.0).with_memory_mb(32_000);
        assert!(matches!(
            s.try_admit(&too_big),
            Err(AdmissionError::Impossible)
        ));
        assert!(!s.is_feasible(&too_big));
    }

    #[test]
    fn lease_release_restores_capacity() {
        let s = scheduler();
        let req = ResourceSpec::cpus(4.0).with_gpus(1.0).with_memory_mb(8000);
        {
            let _full = s.try_admit(&req).unwrap();
            // gpu:0 is now fully used.
            assert!(s.used("gpu:0").unwrap().gpus >= 1.0 - 1e-9);
        }
        // After the lease drops, the device is free again.
        assert!(s.used("gpu:0").unwrap().gpus < 1e-9);
    }

    #[test]
    fn shared_fractional_gpu_packs_onto_one_device() {
        // Three 0.3-GPU trials should share a single 1.0-GPU device (best-fit).
        let s = DeviceScheduler::new([Device::new(
            "gpu:0",
            ResourceSpec::cpus(8.0).with_gpus(1.0).with_memory_mb(16000),
        )]);
        let req = ResourceSpec::cpus(1.0).with_gpus(0.3).with_memory_mb(1000);
        let _a = s.try_admit(&req).unwrap();
        let _b = s.try_admit(&req).unwrap();
        let _c = s.try_admit(&req).unwrap();
        assert!((s.used("gpu:0").unwrap().gpus - 0.9).abs() < 1e-9);
    }
}
