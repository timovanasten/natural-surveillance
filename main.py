import warnings
import logging
import time
from datetime import timedelta, datetime
from config import settings
from pathlib import Path
from analysis.experiments import Experiment
log = logging.getLogger(__name__)


def main():
    log.info("Starting...")
    start_time = time.time()

    # Neighborhoods in Amsterdam to include in the experiment
    neighborhood_names = [
        "Burgwallen-Oude Zijde",
        "Burgwallen-Nieuwe Zijde",
        "Slotermeer-Zuidwest 1",
        "Slotermeer-Noordoost",
        "Osdorp-Oost",
        "De Punt",
        "Slotermeer-Zuidwest 2",
        "Osdorp-Midden",
        "Eendracht",
        "Da Costabuurt",
        "Oude Pijp",
        "Zuid Pijp",
        "Landlust",
        "Van Lennepbuurt",
        "Overtoomse Veld",
        "Van Galenbuurt",
        "Hoofdweg e.o.",
        "Weesperzijde",
        "Grachtengordel-Zuid",
        "De Weteringschans",
        "Kinkerbuurt",
        "De Kolenkit",
        "Jordaan",
        "Nieuwmarkt/Lastage",
        "Nieuwe Pijp",
        "Erasmuspark",
        "Helmersbuurt/Vondelbuurt",
        "Frederik Hendrikbuurt/Centrale Markt",
        "IJselbuurt",
        "Slotervaart Zuid",
        "Chassébuurt",
        "Slotervaart Noord",
        "Geuzenbuurt",
        "Grachtengordel-West",
        "Museumkwartier",
        "Westindische Buurt",
        "Staatsliedenbuurt",
        "Willemspark",
        "Overtoomse Sluis",
        "Rijnbuurt",
        "Schinkelbuurt",
        "Scheldebuurt",
        "Buitenveldert-West",
        ]

    # Retrieve data for the neighborhoods above
    experiment = Experiment(neighborhood_names)
    experiment.collect_data(load_method='none')
    experiment.merge_data_files()

    # Calculate correlations with the Amsterdam Safety Index
    experiment.calculate_correlations(2019)

    end_time = time.time()
    total_runtime_seconds = round(end_time - start_time)
    log.info("Total time: %s", str(timedelta(seconds=total_runtime_seconds)))


if __name__ == '__main__':
    # TODO: Check if I can reset the index of sightline gdf to improve performance
    # Silence the warning from swifter: This pandas object has duplicate indices, and swifter may not
    # be able to improve performance. Consider resetting the indices with `df.reset_index(drop=True)`.
    warnings.filterwarnings('ignore', "/duplicate indices/")
    # For more log options see https://docs.python.org/3/library/logging.html#logrecord-attributes
    handlers = [logging.StreamHandler()]
    if settings.LOG_TO_FILE:
        log_path = Path(settings.LOGGING_DIR, str(datetime.now()) + '.log')
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(format='%(levelname)s - %(module)s: %(message)s',
                        level=logging.INFO,
                        handlers=handlers)
    main()
