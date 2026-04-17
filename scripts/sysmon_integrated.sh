#!/usr/bin/env bash
# =============================================================================
# sysmon_integrated.sh — 系统级性能指标集成监控脚本
# =============================================================================
# 功能：
#   1. 实时（1秒周期）在屏幕上展示：时间、各核CPU利用率、内存、USB中断、
#      温度、磁盘IO、系统负载
#   2. 同时将结构化指标每秒写入 CSV 日志文件（/tmp/sysmon_YYYYMMDD_HHMMSS.csv）
#   3. Ctrl+C 优雅退出，打印日志路径
#
# 依赖（均为标准 Linux 工具，无需 sysstat/Python）：
#   awk, grep, cat, tput, date, nproc
#   /proc/stat, /proc/meminfo, /proc/diskstats, /proc/interrupts, /proc/loadavg
#   /sys/class/thermal/thermal_zone*/temp（可选，自动探测）
#
# 用法：
#   bash sysmon_integrated.sh [LOG_DIR]
#   LOG_DIR 默认为 /tmp
# =============================================================================

# 注意：监控脚本刻意不使用 set -euo pipefail，
# 以确保某个指标采集失败时脚本继续运行而非整体退出。
# 各函数内部已做空值/异常安全处理。

# ---------------------------------------------------------------------------
# 0. 初始化 — 日志路径、ANSI 颜色常量、信号处理
# ---------------------------------------------------------------------------

LOG_DIR="${1:-/tmp}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/sysmon_${TIMESTAMP}.csv"

# ANSI 颜色常量
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
RESET='\033[0m'

# 临时文件（保存上一轮 /proc/stat 和 /proc/diskstats 快照，用于计算增量）
CPU_SNAP="/tmp/.sysmon_cpu_snap_$$"
DISK_SNAP="/tmp/.sysmon_disk_snap_$$"

# Ctrl+C 优雅退出：恢复光标、清理临时文件、打印日志路径
cleanup() {
    tput cnorm 2>/dev/null || true
    rm -f "$CPU_SNAP" "$DISK_SNAP"
    echo ""
    echo -e "${GREEN}[sysmon] 监控已停止。日志文件：${BOLD}${LOG_FILE}${RESET}"
    exit 0
}
trap cleanup SIGINT SIGTERM

# ---------------------------------------------------------------------------
# 1. CSV 日志文件初始化 — 动态生成表头（根据实际 CPU 核心数）
# ---------------------------------------------------------------------------

write_csv_header() {
    local ncpu
    ncpu=$(nproc 2>/dev/null || grep -c '^processor' /proc/cpuinfo 2>/dev/null || echo 1)

    local cpu_cols=""
    for (( i=0; i<ncpu; i++ )); do
        cpu_cols+="cpu${i}_user%,cpu${i}_sys%,cpu${i}_idle%,"
    done
    cpu_cols="${cpu_cols%,}"   # 去掉末尾逗号

    echo "timestamp,${cpu_cols},mem_used_kb,mem_total_kb,disk_r_s,disk_w_s,usb_interrupts,temperatures,loadavg_1m,loadavg_5m,loadavg_15m" \
        > "${LOG_FILE}"
}

# ---------------------------------------------------------------------------
# 2. 指标采集函数（各函数独立，失败时返回安全默认值）
# ---------------------------------------------------------------------------

# 2a. 每核 CPU 利用率（基于 /proc/stat 增量，零延迟）
# 返回格式：空格分隔的 "usr,sys,idle" 字符串，每个元素对应一个核心
# 原理：读取两次 /proc/stat 快照，计算 1 秒内各核增量
get_cpu_stats() {
    local stat_file="/proc/stat"
    [[ ! -r "$stat_file" ]] && echo "0,0,100" && return

    # 读取当前快照（仅取 cpu[0-9]+ 行，每行：cpuN user nice sys idle iowait irq softirq ...）
    local cur
    cur=$(grep -E '^cpu[0-9]' "$stat_file" 2>/dev/null) || cur=""
    if [[ -z "$cur" ]]; then
        echo "0,0,100"
        return
    fi

    if [[ ! -f "$CPU_SNAP" ]]; then
        # 首次调用：保存快照，返回全核 "0,0,100"
        echo "$cur" > "$CPU_SNAP"
        local ncpu
        ncpu=$(echo "$cur" | wc -l)
        local result=""
        for (( i=0; i<ncpu; i++ )); do result+="0,0,100 "; done
        echo "${result% }"
        return
    fi

    local prev
    prev=$(cat "$CPU_SNAP" 2>/dev/null) || prev=""
    echo "$cur" > "$CPU_SNAP"

    # 用 awk 计算增量百分比：每行输出 "usr,sys,idle"（保留1位小数）
    awk -v prev_text="$prev" '
        BEGIN {
            # 解析上一次快照到关联数组
            n = split(prev_text, plines, "\n")
            for (i = 1; i <= n; i++) {
                if (plines[i] == "") continue
                split(plines[i], f)
                cpu_name = f[1]
                p_usr[cpu_name]   = f[2] + f[3]         # user + nice
                p_sys[cpu_name]   = f[4]                 # system
                p_idle[cpu_name]  = f[5]                 # idle
                p_iowait[cpu_name]= f[6]                 # iowait
                p_irq[cpu_name]   = f[7] + f[8]          # irq + softirq
                p_total[cpu_name] = f[2]+f[3]+f[4]+f[5]+f[6]+f[7]+f[8]
            }
        }
        /^cpu[0-9]/ {
            cpu_name = $1
            c_usr   = $2 + $3
            c_sys   = $4
            c_idle  = $5
            c_total = $2+$3+$4+$5+$6+$7+$8

            delta = c_total - p_total[cpu_name]
            if (delta <= 0) delta = 1

            pct_usr  = (c_usr  - p_usr[cpu_name])  * 100 / delta
            pct_sys  = (c_sys  - p_sys[cpu_name])  * 100 / delta
            pct_idle = (c_idle - p_idle[cpu_name]) * 100 / delta

            if (pct_usr  < 0) pct_usr  = 0
            if (pct_sys  < 0) pct_sys  = 0
            if (pct_idle < 0) pct_idle = 0
            if (pct_idle > 100) pct_idle = 100

            printf "%.1f,%.1f,%.1f ", pct_usr, pct_sys, pct_idle
        }
    ' <<< "$cur" | sed 's/ $//'
}

# 2b. 内存信息（直接读 /proc/meminfo，KB）
# 返回："used_kb total_kb"
get_mem_stats() {
    local meminfo="/proc/meminfo"
    [[ ! -r "$meminfo" ]] && echo "0 0" && return

    local total free_kb buffers cached sreclaim
    total=$(awk    '/^MemTotal:/{print $2}'      "$meminfo")
    free_kb=$(awk  '/^MemFree:/{print $2}'       "$meminfo")
    buffers=$(awk  '/^Buffers:/{print $2}'       "$meminfo")
    cached=$(awk   '/^Cached:/{print $2}'        "$meminfo")
    sreclaim=$(awk '/^SReclaimable:/{print $2}'  "$meminfo")

    # 与 `free` 命令口径一致
    local used=$(( ${total:-0} - ${free_kb:-0} - ${buffers:-0} - ${cached:-0} - ${sreclaim:-0} ))
    (( used < 0 )) && used=0
    echo "${used} ${total:-0}"
}

# 2c. 磁盘 IO（基于 /proc/diskstats 增量，单位 ops/s）
# 返回："r_s w_s"
get_disk_io() {
    local diskstats="/proc/diskstats"
    [[ ! -r "$diskstats" ]] && echo "0.00 0.00" && return

    # 过滤物理磁盘（sda/sdb, nvme0n1, vda 等），排除分区（以数字结尾的设备）
    local cur
    cur=$(awk '$3 ~ /^(sd[a-z]|nvme[0-9]+n[0-9]+|vd[a-z]|xvd[a-z]|hd[a-z])$/ {print $3,$4,$8}' \
        "$diskstats" 2>/dev/null) || cur=""

    if [[ -z "$cur" ]]; then
        echo "0.00 0.00"
        return
    fi

    if [[ ! -f "$DISK_SNAP" ]]; then
        echo "$cur" > "$DISK_SNAP"
        echo "0.00 0.00"
        return
    fi

    local prev
    prev=$(cat "$DISK_SNAP" 2>/dev/null) || prev=""
    echo "$cur" > "$DISK_SNAP"

    # fields saved: devname reads_completed writes_completed
    awk -v prev_text="$prev" '
        BEGIN {
            n = split(prev_text, plines, "\n")
            for (i = 1; i <= n; i++) {
                if (plines[i] == "") continue
                split(plines[i], f)
                p_reads[f[1]]  = f[2]+0
                p_writes[f[1]] = f[3]+0
            }
        }
        {
            dev    = $1
            reads  = $2 + 0
            writes = $3 + 0
            dr = reads  - (p_reads[dev]+0)
            dw = writes - (p_writes[dev]+0)
            if (dr < 0) dr = 0
            if (dw < 0) dw = 0
            total_r += dr
            total_w += dw
        }
        END { printf "%.2f %.2f\n", total_r, total_w }
    ' <<< "$cur"
}

# 2d. USB 中断累计次数（读 /proc/interrupts，对 xhci/ehci/uhci/ohci 行求和）
get_usb_interrupts() {
    local val=0
    if [[ -r /proc/interrupts ]]; then
        val=$(grep -iE 'xhci|ehci|uhci|ohci' /proc/interrupts 2>/dev/null \
              | awk '{for(i=2;i<=NF;i++) if($i~/^[0-9]+$/) sum+=$i} END{print sum+0}') || val=0
    fi
    echo "${val:-0}"
}

# 2e. 温度传感器（自动探测所有 thermal_zone，°C）
# 返回：分号分隔的 "zone_type:温度°C" 字符串；无传感器时返回 "N/A"
get_temperatures() {
    local result=""
    # 使用通配符展开；若无匹配，zone_dirs[0] 会是字面字符串
    local -a zone_dirs=(/sys/class/thermal/thermal_zone*)
    if [[ ! -e "${zone_dirs[0]}" ]]; then
        echo "N/A"
        return
    fi

    local zone_dir raw temp_c zone_type
    for zone_dir in "${zone_dirs[@]}"; do
        [[ -r "${zone_dir}/temp" ]] || continue
        raw=$(cat "${zone_dir}/temp" 2>/dev/null)
        # 安全检查：必须是纯数字
        [[ -z "$raw" || ! "$raw" =~ ^[0-9]+$ ]] && continue
        temp_c=$(awk "BEGIN{printf \"%.1f\", ${raw}/1000}")
        zone_type="unknown"
        [[ -r "${zone_dir}/type" ]] && zone_type=$(cat "${zone_dir}/type" 2>/dev/null || echo "unknown")
        result+="${zone_type}:${temp_c}°C;"
    done
    echo "${result%;}"   # 去掉末尾分号；若全部跳过则输出空字符串
}

# 2f. 系统负载（/proc/loadavg）
# 返回："1m 5m 15m"
get_loadavg() {
    local la
    la=$(cat /proc/loadavg 2>/dev/null) || la="0.00 0.00 0.00 0/0 0"
    awk '{print $1, $2, $3}' <<< "$la"
}

# ---------------------------------------------------------------------------
# 3. 屏幕渲染函数
# ---------------------------------------------------------------------------

# 打印彩色分隔线（自适应终端宽度）
print_separator() {
    local char="${1:--}"
    local color="${2:-$CYAN}"
    local width
    width=$(tput cols 2>/dev/null || echo 80)
    echo -e "${color}$(printf '%*s' "$width" | tr ' ' "$char")${RESET}"
}

# 渲染 CPU 栏（每行最多 2 核，含颜色负载警示）
render_cpu_block() {
    local cpu_data="$1"   # 空格分隔的 "usr,sys,idle" 列表
    echo -e "${BOLD}${BLUE}[CPU 核心利用率]${RESET}"

    if [[ -z "$cpu_data" ]]; then
        echo "  (数据采集中…)"
        return
    fi

    local idx=0
    local line_buf=""
    for core_stat in $cpu_data; do
        local usr sys idle
        IFS=',' read -r usr sys idle <<< "$core_stat"
        # 防御性清理，确保是数字
        usr="${usr//[^0-9.]/}";  usr="${usr:-0}"
        sys="${sys//[^0-9.]/}";  sys="${sys:-0}"
        idle="${idle//[^0-9.]/}"; idle="${idle:-0}"

        # 根据利用率着色
        local color=$GREEN
        local used
        used=$(awk "BEGIN{printf \"%d\", int(${usr}+${sys}+0.5)}" 2>/dev/null || echo 0)
        (( used >= 80 )) && color=$RED
        (( used >= 50 && used < 80 )) && color=$YELLOW

        line_buf+=$(printf "  ${color}CPU%-2d${RESET}: usr=%5s%% sys=%5s%% idle=%5s%%" \
            "$idx" "$usr" "$sys" "$idle")

        idx=$(( idx + 1 ))
        # 每 2 核换一行（终端宽度有限时更易读）
        if (( idx % 2 == 0 )); then
            echo -e "$line_buf"
            line_buf=""
        fi
    done
    [[ -n "$line_buf" ]] && echo -e "$line_buf"
}

# 渲染内存栏（含 ASCII 进度条，颜色阈值：60% 黄/85% 红）
render_mem_block() {
    local used_kb="$1"
    local total_kb="$2"

    if [[ -z "$total_kb" || "$total_kb" -eq 0 ]]; then
        echo -e "${BOLD}${BLUE}[内存]${RESET}  N/A"
        return
    fi

    local used_mb total_mb pct bar_len bar_filled bar_empty bar
    used_mb=$(( used_kb / 1024 ))
    total_mb=$(( total_kb / 1024 ))

    pct=$(awk "BEGIN{printf \"%.1f\", ${used_mb}/${total_mb}*100}")
    bar_len=40
    bar_filled=$(awk "BEGIN{printf \"%d\", ${used_mb}/${total_mb}*${bar_len}}")
    bar_filled=$(( bar_filled < 0 ? 0 : (bar_filled > bar_len ? bar_len : bar_filled) ))
    bar_empty=$(( bar_len - bar_filled ))
    bar="$(printf '%*s' "$bar_filled" | tr ' ' '#')$(printf '%*s' "$bar_empty" | tr ' ' '-')"

    local color=$GREEN
    local pct_int="${pct%.*}"
    (( pct_int >= 85 )) && color=$RED
    (( pct_int >= 60 && pct_int < 85 )) && color=$YELLOW

    echo -e "${BOLD}${BLUE}[内存]${RESET}  ${color}[${bar}]${RESET}  ${used_mb} MiB / ${total_mb} MiB  (${pct}%)"
}

# 渲染温度栏（颜色阈值：70°C 黄/85°C 红，每行最多 3 个）
render_temp_block() {
    local temps="$1"
    echo -e "${BOLD}${BLUE}[温度]${RESET}"
    if [[ -z "$temps" || "$temps" == "N/A" ]]; then
        echo "  (无温度传感器数据)"
        return
    fi

    local col=0
    local line_buf=""
    local entry zone_name temp_val val_num val_int color
    IFS=';' read -ra entries <<< "$temps"
    for entry in "${entries[@]}"; do
        [[ -z "$entry" ]] && continue
        zone_name="${entry%%:*}"
        temp_val="${entry##*:}"
        val_num="${temp_val//[^0-9.]/}"
        val_int="${val_num%.*}"
        color=$GREEN
        (( val_int >= 85 )) && color=$RED
        (( val_int >= 70 && val_int < 85 )) && color=$YELLOW

        line_buf+=$(printf "  %-20s ${color}%s${RESET}" "${zone_name}" "${temp_val}")
        col=$(( col + 1 ))
        if (( col % 3 == 0 )); then
            echo -e "$line_buf"
            line_buf=""
        fi
    done
    [[ -n "$line_buf" ]] && echo -e "$line_buf"
}

# ---------------------------------------------------------------------------
# 4. 主监控循环
# ---------------------------------------------------------------------------

# 初始化 CSV 日志（写表头）
write_csv_header

# 隐藏光标（cleanup 会恢复）
tput civis 2>/dev/null || true

# 首次清屏
clear

while true; do
    # ── 并发采集所有指标 ──────────────────────────────────────────────────
    local_time=$(date '+%Y-%m-%d %H:%M:%S')
    cpu_stats=$(get_cpu_stats)
    mem_stats=$(get_mem_stats)
    disk_io=$(get_disk_io)
    usb_irq=$(get_usb_interrupts)
    temps=$(get_temperatures)
    loadavg=$(get_loadavg)

    # 拆分复合值
    mem_used_kb=$(awk '{print $1}' <<< "$mem_stats")
    mem_total_kb=$(awk '{print $2}' <<< "$mem_stats")
    disk_r=$(awk '{print $1}' <<< "$disk_io")
    disk_w=$(awk '{print $2}' <<< "$disk_io")
    la1=$(awk '{print $1}' <<< "$loadavg")
    la5=$(awk '{print $2}' <<< "$loadavg")
    la15=$(awk '{print $3}' <<< "$loadavg")

    # ── 屏幕渲染（移到左上角覆盖，实现 watch 效果）─────────────────────
    tput cup 0 0 2>/dev/null || printf '\033[H'

    print_separator "=" "$CYAN"
    echo -e "  ${BOLD}${CYAN}系统性能监控${RESET}  ${YELLOW}${local_time}${RESET}  (Ctrl+C 退出)"
    print_separator "=" "$CYAN"

    echo ""
    render_cpu_block "$cpu_stats"

    echo ""
    render_mem_block "$mem_used_kb" "$mem_total_kb"

    echo ""
    echo -e "${BOLD}${BLUE}[磁盘 IO]${RESET}  读: ${GREEN}${disk_r} r/s${RESET}   写: ${YELLOW}${disk_w} w/s${RESET}"

    echo ""
    echo -e "${BOLD}${BLUE}[USB 中断]${RESET}  ${MAGENTA}${usb_irq}${RESET} 次（累计）"

    echo ""
    render_temp_block "$temps"

    echo ""
    echo -e "${BOLD}${BLUE}[系统负载]${RESET}  1min: ${GREEN}${la1}${RESET}   5min: ${YELLOW}${la5}${RESET}   15min: ${la15}"

    echo ""
    print_separator "-" "$CYAN"
    echo -e "  日志文件：${BOLD}${LOG_FILE}${RESET}"
    print_separator "-" "$CYAN"

    # ── 写入 CSV 日志 ────────────────────────────────────────────────────
    # cpu_stats：空格分隔 → 逗号分隔（展开各核字段）
    cpu_csv="${cpu_stats// /,}"
    # temps：分号分隔 → 竖线分隔（避免破坏 CSV）
    temps_csv="${temps//;/|}"

    printf '%s,%s,%s,%s,%s,%s,%s,"%s",%s,%s,%s\n' \
        "$local_time" "$cpu_csv" \
        "$mem_used_kb" "$mem_total_kb" \
        "$disk_r" "$disk_w" \
        "$usb_irq" "$temps_csv" \
        "$la1" "$la5" "$la15" \
        >> "${LOG_FILE}"

    # ── 等待 1 秒（在后台 sleep，使 Ctrl+C 能立即响应）─────────────────
    sleep 1 &
    SLEEP_PID=$!
    wait "$SLEEP_PID" 2>/dev/null || true
done
