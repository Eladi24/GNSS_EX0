import rinex_parser as rp

if __name__ == "__main__":
    epochs = rp.parse_rinex_file("gnss_log_2026_03_21_17_14_34.26o")
    print(f"Total epochs parsed: {len(epochs)}")
    print(f"First epoch timestamp: {epochs[0]['timestamp']}")
    print(f"Satellites in first epoch: {epochs[0]['num_sats']}")
    print(f"First satellite: {epochs[0]['satellites'][0]}")
    print(f"Last satellite of the last epoch: {epochs[-1]['satellites'][-1]}")