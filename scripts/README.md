# scripts/sysmon_integrated.sh — 系统级性能指标集成监控

## 功能概述

`sysmon_integrated.sh` 将原来分散在多个终端窗口的系统监控任务整合到**单一 Bash 脚本**中，无需 Python 或外部数据库，开箱即用。

### 实时屏幕展示（1 秒刷新）

| 指标 | 数据来源 |
|------|---------|
| 当前时间 | `date` |
| 每核 CPU 利用率 (usr/sys/idle) | `/proc/stat`（两次快照增量，零延迟） |
| 内存使用/总量 | `/proc/meminfo` |
| USB 中断累计次数 | `/proc/interrupts` (xhci/ehci/uhci/ohci) |
| 温度传感器（自动探测所有 zone） | `/sys/class/thermal/thermal_zone*/temp` |
| 磁盘 IO（r/s、w/s 合计） | `/proc/diskstats`（两次快照增量，零延迟） |
| 系统负载 (1/5/15 min) | `/proc/loadavg` |

### CSV 日志落盘（每秒一行）

日志自动保存到 `/tmp/sysmon_YYYYMMDD_HHMMSS.csv`，字段：

```
timestamp, cpu0_user, cpu0_sys, cpu0_idle, [cpu1..N], mem_used_kb, mem_total_kb,
disk_r_s, disk_w_s, usb_interrupts, temperatures, loadavg_1m, loadavg_5m, loadavg_15m
```

---

## 依赖

所有数据源均来自 Linux 内核标准接口，**无需安装任何额外工具包**：

| 工具/文件 | 用途 | 是否必需 |
|-----------|------|---------|
| `/proc/stat` | CPU 每核利用率（增量计算） | 必需 |
| `/proc/meminfo` | 内存使用情况 | 必需 |
| `/proc/diskstats` | 磁盘 IO（增量计算） | 必需 |
| `/proc/interrupts` | USB 中断计数 | 必需 |
| `/proc/loadavg` | 系统负载 | 必需 |
| `/sys/class/thermal/` | 温度传感器（自动探测） | 可选（无则显示 N/A） |
| `awk`, `grep`, `tput` | 文本处理与终端控制 | 必需（通常已预装） |

---

## 用法

```bash
# 默认日志目录 /tmp
bash scripts/sysmon_integrated.sh

# 指定日志目录
bash scripts/sysmon_integrated.sh /var/log/sysmon
```

按 **Ctrl+C** 停止监控，终端会打印最终日志文件的完整路径。

---

## 输出示例

```
══════════════════════════════════════════════════════════════════════════════
  系统性能监控  2026-04-17 10:32:05  (Ctrl+C 退出)
══════════════════════════════════════════════════════════════════════════════

[CPU 核心利用率]
  CPU0 : usr=  3.0% sys=  1.0% idle= 96.0%   CPU1 : usr=  5.0% sys=  2.0% idle= 93.0%
  CPU2 : usr=  2.0% sys=  1.0% idle= 97.0%   CPU3 : usr=  8.0% sys=  3.0% idle= 89.0%

[内存]  ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  8192 MiB / 32768 MiB  (25.0%)

[磁盘 IO]  读: 12.50 r/s   写: 3.20 w/s

[USB 中断]  487 次（累计）

[温度]
  x86_pkg_temp          65.0°C   acpitz                48.0°C

[系统负载]  1min: 0.45   5min: 0.38   15min: 0.32

──────────────────────────────────────────────────────────────────────────────
  日志文件：/tmp/sysmon_20260417_103205.csv
──────────────────────────────────────────────────────────────────────────────
```

---

## 代码结构

```
sysmon_integrated.sh
├── 0. 初始化           — 日志路径、ANSI 颜色常量、Ctrl+C 信号处理
├── 1. CSV 表头初始化   — 动态生成多核列名
├── 2. 指标采集函数
│   ├── get_cpu_stats()      — 每核 CPU 利用率（/proc/stat 增量）
│   ├── get_mem_stats()      — 内存 used/total (KB，/proc/meminfo)
│   ├── get_disk_io()        — 磁盘 r/s w/s（/proc/diskstats 增量）
│   ├── get_usb_interrupts() — USB 中断累计次数（/proc/interrupts）
│   ├── get_temperatures()   — 所有热区温度（/sys/class/thermal）
│   └── get_loadavg()        — 系统负载均值（/proc/loadavg）
├── 3. 屏幕渲染函数
│   ├── print_separator()    — 分隔线
│   ├── render_cpu_block()   — CPU 彩色表格
│   ├── render_mem_block()   — 内存进度条
│   └── render_temp_block()  — 温度列表
└── 4. 主循环           — 采集→渲染→写CSV→sleep 1s
```

---

## 扩展指南

- **新增指标**：在第 3 节添加采集函数，第 4 节添加渲染函数，主循环中调用，并在 `write_csv_header()` 及 CSV 写入行中添加对应字段。
- **调整刷新频率**：修改主循环末尾 `sleep 1` 的秒数。
- **修改日志格式**：修改 `write_csv_header()` 和主循环内的 `echo "..."` 日志写入行。
- **告警功能**：在渲染函数或主循环中对指标阈值判断，调用 `logger` 或发送通知。
