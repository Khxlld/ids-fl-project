# Device-inspired client profiles

These are **device-inspired Docker resource profiles**, not hardware emulations. The real products motivate a defensible range of edge roles; the Docker limits shape x86 Linux processes so the six-service demo remains stable on a 16-core, 5.7 GiB Docker Desktop VM.

## Real-device evidence and assigned limits

| Client | Logical rows | Real-device inspiration | Manufacturer specification used | Docker CPU | Docker memory |
|---|---:|---|---|---:|---:|
| uav-client-1 | 1,230 | Raspberry Pi 4 Model B | Quad Cortex-A72; 1/2/4/8 GB variants | 0.65 | 512 MiB |
| uav-client-2 | 1,046 | NVIDIA Jetson Nano | Quad Cortex-A57 at up to 1.43 GHz; 4 GB LPDDR4 | 0.55 | 512 MiB |
| uav-client-3 | 591 | Raspberry Pi Zero 2 W | Quad Cortex-A53 at 1 GHz; 512 MB LPDDR2 | 0.35 | 448 MiB |
| uav-client-4 | 2,027 | NVIDIA Jetson Orin Nano 8GB | Six-core Cortex-A78AE; 8 GB module | 1.00 | 768 MiB |
| uav-client-5 | 1,254 | NXP NavQPlus | i.MX 8M Plus companion computer, Dronecode connectors, 8 GB LPDDR4 | 0.45 | 512 MiB |

Authoritative sources:

- [Raspberry Pi 4 Model B product brief](https://datasheets.raspberrypi.com/rpi4/raspberry-pi-4-product-brief.pdf)
- [Raspberry Pi Zero 2 W product brief](https://datasheets.raspberrypi.com/rpizero2/raspberry-pi-zero-2-w-product-brief.pdf)
- [NVIDIA Jetson Nano technical specifications](https://developer.nvidia.com/embedded/jetson-nano)
- [NVIDIA Jetson Orin Nano developer-kit specifications](https://developer.nvidia.com/blog/develop-ai-powered-robots-smart-vision-systems-and-more-with-nvidia-jetson-orin-nano-developer-kit/)
- [NXP NavQPlus companion-computer product page](https://www.nxp.com/design/design-center/development-boards-and-designs/8MPNAVQ)
- [NXP i.MX 8M Plus processor specifications](https://www.nxp.com/products/i.MX8MPLUS)

The Zero 2 W represents a small, memory-constrained edge node. Pi 4 and Jetson Nano represent common general-purpose and AI-IoT companion computers. Orin Nano is the strongest client and receives the largest partition. NavQPlus is especially defensible for UAV discussion because NXP positions it as a mission/companion computer and provides Dronecode-oriented connectors.

## Limits versus hardware

The initial and final limits are identical because the measured run stayed stable: clients peaked near 269-272 MiB and the server near 282 MiB. The configured caps total 3,520 MiB (2,752 MiB clients plus 768 MiB server), leaving roughly 2.3 GiB of the Docker VM for the engine, shared pages, and operating-system headroom. CPU caps total 4.25 host cores.

The limits intentionally do not equal the devices' full RAM or core counts. They constrain one small training process, while real boards must also reserve resources for Linux, flight/robotics software, sensors, and other services.

Docker Desktop on x86 can demonstrate scheduling pressure, memory ceilings, non-IID data sizes, and visible stragglers. It does **not** emulate ARM instruction sets, CPU microarchitecture, GPU/NPU acceleration, power draw, thermal throttling, real radio links, sensor I/O, flight dynamics, or actual on-device latency. No hardware-performance or energy claim should be made from these measurements.
