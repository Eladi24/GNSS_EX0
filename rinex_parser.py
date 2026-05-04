from datetime import datetime

def parse_rinex_file(file_path) -> list:
    
    epochs = []
    obs_types = {}
    header_lines = []
    current_epoch = None
    in_header = True
    with open(file_path, 'r') as rinex_file:
        for line in rinex_file:
            if in_header:
                header_lines.append(line)
                if "END OF HEADER" in line:
                    obs_types = parse_obs_types(header_lines)
                    in_header = False
            else:
                if line.startswith(">"):
                    if current_epoch is not None:
                        epochs.append(current_epoch)
                    time_stamp, epoch_flag, num_sats = parse_epoch_header(line)
                    # RINEX epoch flag 0 is OK, 1 is power failure (contains valid data).
                    if epoch_flag > 1:
                        current_epoch = None
                        continue
                    current_epoch = {
                        "timestamp": time_stamp,
                        "num_sats": num_sats,
                        "satellites": []
                    }
                else:
                    if current_epoch is None:
                        continue
                    sat = parse_sat_line(line, obs_types)
                    if sat is not None:
                        current_epoch["satellites"].append(sat)
        if current_epoch is not None:
            epochs.append(current_epoch)
    return epochs

def parse_sat_line(line, obs_types) -> dict:

    sattelite_id = line[0:3].strip()
    constellation_letter = sattelite_id[0]
    obs_list = obs_types.get(constellation_letter, [])
    prn = int(sattelite_id[1:])
    
    # obs_list is [] if the constellation is unsupported, not None.
    if not obs_list:
        return None
    sattelite_data = {
        "id": sattelite_id,
        "sys": constellation_letter,
        "prn": prn
    }

    has_relevant_data = False
    for n, obs_name in enumerate(obs_list):
        # RINEX 3 data fields are 14 chars wide, starting at index 4 for the first obs
        # (A3, 1X, F14.3, I1, I1). Previous offset of 3 caused an off-by-one error
        # which accidentally included the previous observation's SSI flag.
        start = 4 + (n * 16)
        end = start + 14

        raw = line[start:end] if end <= len(line) else ""
        try:
            val = float(raw.split()[0]) if raw.strip() else None
        except ValueError:
            val = None
        sattelite_data[obs_name] = val
        if val is not None:
            has_relevant_data = True
            
    if not has_relevant_data:
        return None
        
    return sattelite_data


def parse_epoch_header(line) -> tuple[datetime, int, int]:

    parts = line.split()
    year = int(parts[1])
    month = int(parts[2])
    day = int(parts[3])
    hour = int(parts[4])
    minute = int(parts[5])
    second = float(parts[6])
    whole_second = int(second)
    microsecond = int((second - whole_second) * 1e6)
    time_stamp = datetime(year, month, day, hour, minute, whole_second, microsecond)
    epoch_flag = int(parts[7])
    num_sats = int(parts[8])
    return time_stamp, epoch_flag, num_sats

def parse_obs_types(header_lines) -> dict:
    
    obs_types = {}
    for line in header_lines:
        if "SYS / # / OBS TYPES" in line:
            constellation_letter = line[0]
            num_types = int(line[3:6].strip())
            types = []
            for i in range(num_types):
                start = 7 + (i * 4)
                end = start + 3
                types.append(line[start:end].strip())
            obs_types[constellation_letter] = types
    return obs_types