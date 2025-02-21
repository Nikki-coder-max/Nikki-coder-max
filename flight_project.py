import os
import json
import random
import time
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
#
#/tmp/flights/%MM-YY%-%origin_city%-flights.json
#N=~5000 JSON files
#Total set of cities is K=[100-200]
#where each file is a JSON array of random size M = [50 – 100] of randomly generated flights data between cities.
# Constants
N_FILES = 5000  # Number of JSON files N=~5000 JSON 
CITY_COUNT = random.randint(100, 200)  # Total unique cities
FLIGHTS_PER_FILE = (50, 100)  # Random flight count per file
NULL_PROBABILITY = random.uniform(0.005, 0.001)  # 0.5% - 0.1% probability
FLIGHT_DIR = "/tmp/flights/"

# Generate Random Cities
CITIES = [f"City-{i}" for i in range(CITY_COUNT)]

# Ensure directory exists
os.makedirs(FLIGHT_DIR, exist_ok=True)
def generate_flight_data():
    """Generate random flight data with occasional NULL values."""
    return {
        "date": (datetime.now() - timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d") if random.random() > NULL_PROBABILITY else None,
        "origin_city": random.choice(CITIES),
        "destination_city": random.choice(CITIES),
        "flight_duration_secs": random.randint(1800, 43200) if random.random() > NULL_PROBABILITY else None,  # Between 30 min - 12 hours
        "passengers_on_board": random.randint(1, 500) if random.random() > NULL_PROBABILITY else None,  # 1-500 passengers
    }


def generate_files():
    """Generate N JSON files with random flight data."""
    for _ in range(N_FILES):
        origin_city = random.choice(CITIES)
        filename = os.path.join(FLIGHT_DIR, f"{datetime.now().strftime('%m-%y')}-{origin_city}-flights.json")

        flights = [generate_flight_data() for _ in range(random.randint(*FLIGHTS_PER_FILE))]
        
        with open(filename, "w") as f:
            json.dump(flights, f)

    print(f" {N_FILES} flight data files generated in {FLIGHT_DIR}")

def process_file(filename):
    """Process a single JSON file, returning aggregated statistics."""
    try:
        with open(filename, "r") as f:
            flights = json.load(f)
    except Exception:
        return None  # Skip if file is corrupted

    record_count = 0
    dirty_count = 0
    duration_data = defaultdict(list)
    passenger_count = defaultdict(int)

    for flight in flights:
        record_count += 1

        # Check if any field is None
        if any(v is None for v in flight.values()):
            dirty_count += 1
            continue  # Skip dirty records

        # Aggregate flight durations for destination cities
        destination = flight["destination_city"]
        duration_data[destination].append(flight["flight_duration_secs"])

        # Track passenger movements
        passenger_count[flight["origin_city"]] -= flight["passengers_on_board"]
        passenger_count[destination] += flight["passengers_on_board"]

    return record_count, dirty_count, duration_data, passenger_count


def analyze_data():
    """Analyze and clean the generated flight data."""
    start_time = time.time()

    # Get all JSON files
    json_files = [os.path.join(FLIGHT_DIR, f) for f in os.listdir(FLIGHT_DIR) if f.endswith(".json")]

    # Use multiprocessing for faster file processing
    with Pool(cpu_count()) as pool:
        results = pool.map(process_file, json_files)

    # Aggregate results
    total_records, total_dirty = 0, 0
    aggregated_durations = defaultdict(list)
    aggregated_passengers = defaultdict(int)

    for result in results:
        if result is None:
            continue  # Skip if file couldn't be read

        records, dirty, durations, passengers = result
        total_records += records
        total_dirty += dirty

        # Merge duration data
        for city, times in durations.items():
            aggregated_durations[city].extend(times)

        # Merge passenger counts
        for city, count in passengers.items():
            aggregated_passengers[city] += count

    # Compute Top 25 destination cities by flight count
    top_25_destinations = sorted(aggregated_durations.keys(), key=lambda city: len(aggregated_durations[city]), reverse=True)[:25]

    # Compute AVG & 95th percentile flight duration
    duration_stats = {}
    for city in top_25_destinations:
        durations = aggregated_durations[city]
        if durations:
            duration_stats[city] = {
                "AVG_duration_secs": round(np.mean(durations), 2),
                "P95_duration_secs": round(np.percentile(durations, 95), 2),
            }

    # Identify cities with max passengers arrived & left
    max_arrivals = max(aggregated_passengers.items(), key=lambda x: x[1])
    max_departures = min(aggregated_passengers.items(), key=lambda x: x[1])

    end_time = time.time()

    # Print Analysis Results
    print("**Analysis Report**")
    print(f"Total Records Processed: {total_records}")
    print(f" Dirty Records Found: {total_dirty}")
    print(f" Total Run Duration: {round(end_time - start_time, 2)} seconds")
    print(" **Top 25 Destination Cities (Flight Duration Stats)**")
    for city, stats in duration_stats.items():
        print(f"{city}: AVG = {stats['AVG_duration_secs']}s, P95 = {stats['P95_duration_secs']}s")

    print(f" City with MAX Arrivals: {max_arrivals[0]} ({max_arrivals[1]} passengers)")
    print(f" City with MAX Departures: {max_departures[0]} ({abs(max_departures[1])} passengers)")

if __name__ == "__main__":
    print("Generating Flight Data...")
    generate_files()

    print(" Analyzing Flight Data...")
    analyze_data()
