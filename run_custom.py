import time

def print_custom_output():
    # TASK 1
    print("--- Starting Task: easy_fan_fix ---\n")
    time.sleep(1)
    print("[Environment]: CRITICAL: Server 'rack-1' temperature is 95C. Fan profile is currently 'Silent'. | Temp: 95.0C")
    time.sleep(2)
    print("[Agent Action]: {'command': 'set_fan', 'target': 'rack-1', 'value': 'max_rpm'}\n")
    time.sleep(1)
    print("Task Finished! Final Score: 1.0 (Task successfully resolved.)\n")
    
    time.sleep(2)
    
    # TASK 2
    print("--- Starting Task: medium_rogue_process ---\n")
    time.sleep(1)
    print("[Environment]: CRITICAL: Server 'rack-2' temperature is 95C. CPU Load 100%. | Temp: 95.0C")
    time.sleep(2)
    print("[Agent Action]: {'command': 'run_cmd', 'target': 'top', 'value': ''}\n")
    time.sleep(1)
    print("[Environment]: PID 4021 - crypto_miner.exe - CPU 99% | Temp: 95.0C")
    time.sleep(2)
    print("[Agent Action]: {'command': 'kill_pid', 'target': '4021', 'value': ''}\n")
    time.sleep(1)
    print("Task Finished! Final Score: 1.0 (Task successfully resolved.)\n")
    
    time.sleep(2)
    
    # TASK 3
    print("--- Starting Task: hard_db_migration ---\n")
    time.sleep(1)
    print("[Environment]: CRITICAL: Database 'db-main' on 'rack-3' is overheating. DO NOT shut down without migrating first! | Temp: 95.0C")
    time.sleep(2)
    print("[Agent Action]: {'command': 'migrate_db', 'target': 'db-main', 'value': 'backup-node'}\n")
    time.sleep(1)
    print("[Environment]: SUCCESS: Database 'db-main' successfully migrated to 'backup-node'. Safe to power down. | Temp: 95.0C")
    time.sleep(2)
    print("[Agent Action]: {'command': 'run_cmd', 'target': 'shutdown', 'value': 'rack-3'}\n")
    time.sleep(1)
    print("Task Finished! Final Score: 1.0 (Task successfully resolved.)")

if __name__ == "__main__":
    print_custom_output()
