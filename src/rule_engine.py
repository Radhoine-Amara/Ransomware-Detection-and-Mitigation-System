"""
Rule-Based Detection Engine for ransome 
->input systemEvent 
->output ALert 
"""

import json
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional


#  LOGGING SETUP into console + log file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),                          # console
        logging.FileHandler("project_alerts.log", mode="a") # log file
    ]
)
logger = logging.getLogger("RuleEngine")

#  DATA STRUCTURES
@dataclass # input is the systemevent 
class SystemEvent:# here it is like a report of something that happened on the computer 
    
    # Process info
    process_name: str = ""
    process_args: str = ""
    parent_process: str = "" # here it responds to what program ran / whta command did it execute ....
    pid: int = 0

    # File system activity
    files_renamed_last_10s: int = 0
    files_written_last_30s: int = 0
    avg_file_entropy: float = 0.0       # Shannon entropy of written files (0.0 - 8.0)

    # Network activity
    outbound_ip: str = ""
    beacon_interval_seconds: float = 0.0  # regularity of outbound requests
    dns_queries_per_minute: int = 0
    dns_target_domain_age_days: int = 999

    # System changes
    new_scheduled_task: bool = False
    new_service_created: bool = False
    created_by_admin: bool = False

    # Memory access
    target_process: str = ""            # process being accessed
    access_type: str = ""               

    # Timestamp
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Alert:
    """
    Generated when a rule is triggered.
    """
    rule_id: str
    severity: str                        # CRITICAL / HIGH / MEDIUM / LOW
    category: str    # when we know the severity we can know what would the response be 
    description: str
    response_actions: list
    event: SystemEvent
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "rule_id": self.rule_id,
            "severity": self.severity,
            "category": self.category,
            "description": self.description,
            "response_actions": self.response_actions,
            "process": self.event.process_name,
            "pid": self.event.pid,
        }

    def log(self):
        level = {
            "CRITICAL": logger.critical,
            "HIGH":     logger.error,
            "MEDIUM":   logger.warning,
            "LOW":      logger.info,
        }.get(self.severity, logger.info)

        level(
            f"[{self.rule_id}] [{self.severity}] {self.description} | "
            f"Process: {self.event.process_name} (PID {self.event.pid}) | "
            f"Actions: {', '.join(self.response_actions)}"
        )


#  RULE ENGINE


class RuleEngine:
    """
    Checks a SystemEvent against all 12 detection rules.
    Returns a list of Alerts triggered (can be more than one per event).
    """
    # checks a systemEvent against all 12 detection rules we put in a rules file and returns a list
    #of alerts triggered can be more than one per event 

    SUSPICIOUS_PARENT_PROCESSES = {"winword.exe", "excel.exe", "powerpnt.exe", "chrome.exe", "firefox.exe", "outlook.exe"}
    SUSPICIOUS_CHILD_PROCESSES  = {"cmd.exe", "powershell.exe", "wscript.exe", "cscript.exe", "mshta.exe"}

    def check(self, event: SystemEvent) -> list[Alert]:
        alerts = []

        for rule_method in [
            self._rule_001_vssadmin,
            self._rule_002_mass_file_rename,
            self._rule_003_lsass_access,
            self._rule_004_c2_beaconing,
            self._rule_005_bcdedit,
            self._rule_006_entropy_spike,
            self._rule_007_suspicious_process_chain,
            self._rule_008_wbadmin,
            self._rule_009_cipher,
            self._rule_010_dns_anomaly,
            self._rule_011_powershell_encoded,
            self._rule_012_persistence,
        ]:
            alert = rule_method(event)
            if alert:
                alert.log()
                alerts.append(alert)

        return alerts

    #RULE 001 
    def _rule_001_vssadmin(self, event: SystemEvent) -> Optional[Alert]:
        
        if (event.process_name.lower() == "vssadmin.exe"
                and "delete" in event.process_args.lower()
                and "shadows" in event.process_args.lower()):
            return Alert(
                rule_id="RULE-001",
                severity="CRITICAL",
                category="Command Execution / Anti-Recovery",
                description="Shadow copy deletion detected — ransomware removing recovery points",
                response_actions=["kill_process", "send_alert", "log_event"],
                event=event,
            )

    #RULE 002 
    def _rule_002_mass_file_rename(self, event: SystemEvent) -> Optional[Alert]:
       
        if event.files_renamed_last_10s > 20:
            return Alert(
                rule_id="RULE-002",
                severity="CRITICAL",
                category="File System Behavior",
                description=f"Mass file renaming detected — {event.files_renamed_last_10s} files renamed in 10s (active encryption likely)",
                response_actions=["suspend_process", "snapshot_directory", "send_alert", "log_event"],
                event=event,
            )

    #RULE 003
    def _rule_003_lsass_access(self, event: SystemEvent) -> Optional[Alert]:
        
        if (event.target_process.lower() == "lsass.exe"
                and "PROCESS_VM_READ" in event.access_type):
            return Alert(
                rule_id="RULE-003",
                severity="HIGH",
                category="Credential Theft / Process Injection",
                description="Suspicious memory access to lsass.exe — possible credential dumping",
                response_actions=["block_memory_access", "send_alert", "log_event"],
                event=event,
            )

    #RULE 004 
    def _rule_004_c2_beaconing(self, event: SystemEvent) -> Optional[Alert]:
        
        if 30 <= event.beacon_interval_seconds <= 300 and event.outbound_ip:
            return Alert(
                rule_id="RULE-004",
                severity="HIGH",
                category="Network / C2",
                description=f"C2 beaconing detected — regular requests to {event.outbound_ip} every {event.beacon_interval_seconds}s",
                response_actions=["block_outbound_connection", "log_c2_ip", "send_alert", "log_event"],
                event=event,
            )

    #RULE 005 ─
    def _rule_005_bcdedit(self, event: SystemEvent) -> Optional[Alert]:
    
        args_lower = event.process_args.lower()
        if (event.process_name.lower() == "bcdedit.exe"
                and ("recoveryenabled" in args_lower or "bootstatuspolicy" in args_lower)):
            return Alert(
                rule_id="RULE-005",
                severity="CRITICAL",
                category="Command Execution / Anti-Recovery",
                description="bcdedit disabling boot recovery — ransomware eliminating recovery options",
                response_actions=["kill_process", "restore_boot_config", "send_alert", "log_event"],
                event=event,
            )

    #RULE 006 
    def _rule_006_entropy_spike(self, event: SystemEvent) -> Optional[Alert]:
        
        if event.avg_file_entropy > 7.2 and event.files_written_last_30s > 10:
            return Alert(
                rule_id="RULE-006",
                severity="HIGH",
                category="File System Behavior / Entropy",
                description=f"Entropy spike detected — avg {event.avg_file_entropy:.2f} bits/byte across {event.files_written_last_30s} files (active encryption likely)",
                response_actions=["send_alert", "begin_file_snapshot", "flag_process", "log_event"],
                event=event,
            )

    #RULE 007
    def _rule_007_suspicious_process_chain(self, event: SystemEvent) -> Optional[Alert]:
        
        if (event.parent_process.lower() in self.SUSPICIOUS_PARENT_PROCESSES
                and event.process_name.lower() in self.SUSPICIOUS_CHILD_PROCESSES):
            return Alert(
                rule_id="RULE-007",
                severity="HIGH",
                category="Process Tree / Initial Access",
                description=f"Suspicious process chain: {event.parent_process} → {event.process_name} — possible macro-based malware",
                response_actions=["kill_child_process", "quarantine_parent_document", "send_alert", "log_event"],
                event=event,
            )

    #RULE 008 
    def _rule_008_wbadmin(self, event: SystemEvent) -> Optional[Alert]:
        
        args_lower = event.process_args.lower()
        if (event.process_name.lower() == "wbadmin.exe"
                and "delete" in args_lower
                and ("catalog" in args_lower or "systemstatebackup" in args_lower)):
            return Alert(
                rule_id="RULE-008",
                severity="CRITICAL",
                category="Command Execution / Anti-Recovery",
                description="wbadmin backup catalog deletion — ransomware destroying Windows backup",
                response_actions=["kill_process", "send_alert", "attempt_backup_restore", "log_event"],
                event=event,
            )

    #RULE 009
    def _rule_009_cipher(self, event: SystemEvent) -> Optional[Alert]:
        
        if (event.process_name.lower() == "cipher.exe"
                and "/w" in event.process_args.lower()):
            return Alert(
                rule_id="RULE-009",
                severity="MEDIUM",
                category="Command Execution / Anti-Forensics",
                description="cipher.exe /w detected — secure wipe of free space, possible anti-forensics",
                response_actions=["send_alert", "log_affected_path", "flag_for_investigation", "log_event"],
                event=event,
            )

    #RULE 010
    def _rule_010_dns_anomaly(self, event: SystemEvent) -> Optional[Alert]:
        
        if event.dns_queries_per_minute > 50 or event.dns_target_domain_age_days < 30:
            reason = (
                f"{event.dns_queries_per_minute} DNS queries/min (DGA suspected)"
                if event.dns_queries_per_minute > 50
                else f"domain age {event.dns_target_domain_age_days} days (newly registered)"
            )
            return Alert(
                rule_id="RULE-010",
                severity="MEDIUM",
                category="Network / DNS Anomaly",
                description=f"Suspicious DNS activity — {reason}",
                response_actions=["send_alert", "log_queried_domains", "flag_process", "log_event"],
                event=event,
            )

    #RULE 011 
    def _rule_011_powershell_encoded(self, event: SystemEvent) -> Optional[Alert]:
    
        args_lower = event.process_args.lower()
        if (event.process_name.lower() == "powershell.exe"
                and ("-encodedcommand" in args_lower or "-enc " in args_lower
                     or ("bypass" in args_lower and "hidden" in args_lower))):
            return Alert(
                rule_id="RULE-011",
                severity="HIGH",
                category="Command Execution / Obfuscation",
                description="PowerShell obfuscated execution detected — encoded or hidden command",
                response_actions=["intercept_command", "decode_and_log", "send_alert", "log_event"],
                event=event,
            )

    #RULE 012
    def _rule_012_persistence(self, event: SystemEvent) -> Optional[Alert]:
        
        if (event.new_scheduled_task or event.new_service_created) and not event.created_by_admin:
            what = "scheduled task" if event.new_scheduled_task else "service"
            return Alert(
                rule_id="RULE-012",
                severity="MEDIUM",
                category="Persistence",
                description=f"New {what} created by non-admin process {event.process_name} — possible persistence mechanism",
                response_actions=["send_alert", "log_task_details", "flag_creating_process", "log_event"],
                event=event,
            )
#  DEMO to test 
if __name__ == "__main__":

    engine = RuleEngine()

    print("\n" + "="*60)
    print("  S004 RuleEngine — Demo Test Run")
    print("="*60 + "\n")

    test_events = [
        # RULE-001: WannaCry deletes shadow copies 
        SystemEvent(process_name="vssadmin.exe", process_args="delete shadows /all /quiet", pid=1234),

        # RULE-002: LockBit encrypting files fast
        SystemEvent(process_name="lockbit.exe", files_renamed_last_10s=45, pid=5678),

        # RULE-003: REvil dumping credentials
        SystemEvent(process_name="malware.exe", target_process="lsass.exe", access_type="PROCESS_VM_READ", pid=9999),

        # RULE-005: Ryuk disabling recovery
        SystemEvent(process_name="bcdedit.exe", process_args="/set {default} recoveryenabled No", pid=3333),

        # RULE-006: Entropy spike during encryption
        SystemEvent(process_name="cryptolocker.exe", avg_file_entropy=7.8, files_written_last_30s=25, pid=4444),

        # RULE-007: Word spawning PowerShell (Emotet)
        SystemEvent(process_name="powershell.exe", parent_process="winword.exe", pid=2222),

        # RULE-011: Encoded PowerShell
        SystemEvent(process_name="powershell.exe", process_args="-EncodedCommand aQBlAHgA -WindowStyle Hidden -Exec Bypass", pid=7777),

        # Clean event — should produce no alerts
        SystemEvent(process_name="notepad.exe", process_args="document.txt", pid=1111),
    ]

    total_alerts = 0
    for i, event in enumerate(test_events, 1):
        alerts = engine.check(event)
        total_alerts += len(alerts)
        if not alerts:
            print(f"[Event {i}] process={event.process_name} → No alert (clean)\n")

    print("\n" + "="*60)
    print(f"  Test complete — {total_alerts} alerts triggered across {len(test_events)} events")
    print(f"  Alerts also saved to: s004_alerts.log")
    print("="*60 + "\n")
