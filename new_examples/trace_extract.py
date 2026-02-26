import json
import configparser
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
from typing import Dict, Any

def parse_duration_npy(data,
                       output_trace: str,
                       loc_probe_map: dict[int, int], 
                       time_unit: str = "cycle",
                       chunk_size: int = 4096,
                       workers: int = 16,
                       queue_size: int = 16) -> Dict[str, Any]:

    if data.size == 0:
        return {"schemaVersion": 1, "traceEvents": [], "displayTimeUnit": time_unit}

    n_thread = data.shape[1] // 4

    # Bounded queue to cap memory used by pending serialized chunks
    q: "queue.Queue[object]" = queue.Queue(maxsize=queue_size)

    def worker(start_row: int, end_row: int):
        parts = []
        for row in data[start_row:end_row]:
            for i in range(n_thread):
                start_index = int(row[i * 4])
                end_index = int(row[i * 4 + 1])
                start_time = int(row[i * 4 + 2])
                end_time = int(row[i * 4 + 3])

                if start_time == 0:
                    continue

                duration = end_time - start_time
                loc = loc_probe_map.get(str(start_index), None)
                if loc is None:
                    # keep behavior minimal: label unknown probes to avoid crashing
                    name = f"Line {start_index}"
                    cat = name
                else:
                    name = f"Line {loc}"
                    cat = f"Line {loc}"

                event = {
                    "name": name,
                    "cat": cat,
                    "ph": "X",
                    "ts": start_time,
                    "dur": duration,
                    "pid": 0,
                    "tid": i,
                    "args": {
                        "start_index": start_index,
                        "end_index": end_index
                    }
                }
                parts.append(json.dumps(event, ensure_ascii=False))

        if parts:
            # join into one string to reduce queue ops and memory overhead
            q.put(',\n'.join(parts))

    def writer_thread_fn():
        first = True
        with open(output_trace, 'w', encoding='utf-8') as f:
            f.write('{' + f'"schemaVersion": 1, "traceEvents": [\n')
            while True:
                item = q.get()
                if item is None:
                    break
                if not first:
                    f.write(',\n')
                f.write(item)
                first = False
            f.write('\n],')
            f.write(f'"displayTimeUnit": "{time_unit}"' + '}')

    total_rows = data.shape[0]

    # Launch writer
    writer = threading.Thread(target=writer_thread_fn, daemon=True)
    writer.start()

    # Submit worker tasks for row ranges
    ranges = []
    for start in range(0, total_rows, chunk_size):
        end = min(start + chunk_size, total_rows)
        ranges.append((start, end))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(worker, s, e) for s, e in ranges]
        # wait for completion
        for fut in futures:
            fut.result()

    # signal writer to finish
    q.put(None)
    writer.join()

    print(f"成功输出Trace文件: {output_trace}")

    return {"schemaVersion": 1, "traceEvents": "streamed", "displayTimeUnit": time_unit}

def main():
    config_path = "/home/junshan/cuTile/config.ini"
    config = configparser.ConfigParser()
    config.read(config_path)

    loc_map = config['output']['loc_map']
    t_start = int(config['ominiscope']['t_start'])
    t_end = int(config['ominiscope']['t_end'])
    input = config['output']['raw_base'] + f"_{t_start}-{t_end}.npy"
    output = config['output']['trace_base'] + f"_{t_start}-{t_end}.json"
    with open(loc_map, 'r', encoding='utf-8') as f:
        map = json.load(f)

    raw_buffer = np.load(input, mmap_mode='r')

    parse_duration_npy(
        data=raw_buffer,
        output_trace=output,
        loc_probe_map=map
    )

if __name__ == '__main__':
    main()
