# disk_maintenance

## check_usage
- df_human: `df -h`
- du_top10: `du -sh /* 2>/dev/null | sort -rh | head -10`

## clean_temp
- clean_tmp: `rm -rf /tmp/* 2>/dev/null; echo "done"`
- clean_logs: `journalctl --vacuum-time=7d`
- clean_apt: `apt-get autoremove -y && apt-get clean`

## check_health
- smart_status: `smartctl -H /dev/sda 2>/dev/null || echo "smartctl not available"`
- inode_usage: `df -i`
- mount_check: `mount | grep -E 'ext4|xfs|btrfs'`
