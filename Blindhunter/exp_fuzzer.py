import config as _Config
import logger as _Logger
import mutator as _Mutator
import parser as _Parser
import numpy as np
import os
import subprocess
import time
import multiprocessing
import hashlib
import json
import argparse
import shutil
import pdb
import carla
import random
import sys

LOG = _Logger.get_logger(_Config.__prog__)



class Scenario:
    def __init__(self, _seed_path):
        self.seed_path = _seed_path
        self.hash = None
        self.verified = False
        self.occ_score = 0.0
        self.score = 0.0
        with open(_seed_path, "r") as file:
            instance = json.load(file)
        self.instance = instance
        self.oracle = False

    def update_instance(self):
        with open(self.seed_path, "r") as file:
            instance = json.load(file)
        self.instance = instance

    def __lt__(self, scene):
        return self.score < scene.score

    def __le__(self, scene):
        return self.score <= scene.score

    def get_hash(self):
        if self.hash is None:
            data_to_hash = json.dumps({
                "seed_path": self.seed_path,
                "score": self.score,
                "verified": self.verified,
                "simtime": time.time()
            }, indent=4).encode()

            self.hash = hashlib.md5(data_to_hash).hexdigest()
        return self.hash

def corpus_parser(_seeds):
    """
    Return a list of scenario objects for json files in the corpus directory
    Args:
        _seeds (dir_path): The corpus directory
    Returns:
        list of scenario objects
    """
    corpus = []
    # for file in seeds: corpus.append(scenario_parse(file))
    for file in os.listdir(_seeds):
        if file.endswith(".json"):
            file_path = os.path.join(_seeds, file)
            with open(file_path, "r") as f:
                seed = json.load(f)
            _scenario = Scenario(file_path)
            corpus.append(_scenario)
    return corpus

class Fuzzer:
    def __init__(self, _corpus, _workdir, _map, _strategy, _desired_occlusion, _carla_path, _tracking_path, _exp_path, _benchmark_path):
        self.carla_process = None
        self.carla_path = _carla_path
        self.tracking_path = _tracking_path
        self.exp_path = _exp_path
        self.benchmark_path = _benchmark_path
        self.corpus = corpus_parser(_corpus)
        self.workdir = _workdir
        self.map = _map
        self.strategy = _strategy
        self.desired_occlusion = _desired_occlusion
        self.init_workdir()
        # the seed queue
        self.population = []
        # seeds that can trigger collision
        self.collisions = []
        # testcases that have already been executed, used for testcase deduplication
        self.hashes = set([])
        self.simtime = 0
        self.oracle = False
        self.client = None
        self.flag = 0
        self.global_cnt = 0
        self.tracking_path = None
        snapshot_path = self.workdir + f"/{str(self.strategy)}/{str(self.desired_occlusion)}/snapshot"
        if os.path.exists(snapshot_path):
            print("reloading snapshot")
            self.load_snapshot()

    def init_workdir(self):
        base_path = os.path.join(self.workdir, str(self.strategy), str(self.desired_occlusion))

        if not os.path.exists(base_path):
            os.makedirs(base_path)

        subdirs = ['queue', 'dataset', 'trace']
        for subdir in subdirs:
            subdir_path = os.path.join(base_path, subdir)
            if not os.path.exists(subdir_path):
                os.mkdir(subdir_path)

        info_file_path = os.path.join(base_path, 'info')
        if not os.path.exists(info_file_path):
            with open(info_file_path, "w") as f:
                pass

        snapshot_file_path = os.path.join(base_path, 'snapshot')
        if not os.path.exists(snapshot_file_path):
            with open(snapshot_file_path, "w") as snapshot:
                pass

    def stop_carla(self):
        try:
            result = subprocess.run(['pkill', '-f', 'CarlaUE4'], check=True)
            print("Existing CARLA processes terminated.")
            return True
        except subprocess.CalledProcessError:
            print("No existing CARLA processes were found.")
            return False

    def run_carla(self):
        try:
            process = subprocess.Popen([self.carla_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.carla_process = process  # Store the process to manage it later
            time.sleep(12)  # Wait for CARLA to start
            if process.poll() is None:  # Process is running
                print("CARLA started successfully.")
                return True
            else:
                print("CARLA failed to start.")
                return False
        except Exception as e:
            print(f"Failed to start CARLA: {str(e)}")
            return False

    def restart_simulation(self):
        """
        This method ensures CARLA is restarted before running any instance.
        """
        print("Stopping existing CARLA processes...")
        if self.stop_carla():
            print("Retrying to terminate CARLA processes...")
        else:
            print("No CARLA processes found, moving to start a new one.")

        print("Starting CARLA simulation...")
        if not self.run_carla():
            self.restart_simulation()
        else:
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(30.0)
            print("CARLA simulation is running.")

    # save the success scenario
    def save_scenario_trace(self, seed):
        if seed.hash is None:
            trace_path = f"{self.workdir}/{str(self.strategy)}/{str(self.desired_occlusion)}/trace/{seed.get_hash()}.json"

        else:
            trace_path = f"{self.workdir}/{str(self.strategy)}/{str(self.desired_occlusion)}/trace/{seed.hash}.json"

        with open(trace_path, "w") as outfile:
            json.dump(seed.instance, outfile, indent=4)

    def save_scenario_queue(self, seed):
        if seed.hash is None:
            trace_path = f"{self.workdir}/{str(self.strategy)}/{str(self.desired_occlusion)}/queue/{seed.get_hash()}.json"

        else:
            trace_path = f"{self.workdir}/{str(self.strategy)}/{str(self.desired_occlusion)}/queue/{seed.hash}.json"
        with open(trace_path, "w") as outfile:
            json.dump(seed.instance, outfile, indent=4)
        return trace_path

    # get rid of unwanted dataset
    def remove_scenario_data(self, seed):
        data_path_dir = f"{self.workdir}/{str(self.strategy)}/{str(self.desired_occlusion)}/dataset/{seed.hash}"
        if os.path.exists(data_path_dir):
            for filename in os.listdir(data_path_dir):
                file_path = os.path.join(data_path_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            os.rmdir(data_path_dir)
        else:
            pass

    def run_instance(self, _seed, _save=True):
        '''
        # restart simulation every 30m
        if time.time() - self.simtime >= 1800:
            LOG.info("Restart Carla simulation")
            self.restart_simulation()
            self.simtime = time.time()

            LOG.info("running seed once to init environment")
            self.run_instance(_seed, _save=False)
        '''
        self.restart_simulation()
        # check if it is a already-verified testcase
        '''
        if _seed.hash in self.hashes:
            LOG.info("Duplicate seed: {}".format(_seed.hash))
            return None, None, None
        '''

        mutator = _Mutator.final_Mutator(_seed.seed_path)

        # check if the seed is initialized
        if _seed.instance['N'] is None:
            try:
                mutator.run_initial(_seed.instance, self.client, self.desired_occlusion)
            except Exception as e:
                print(f"An error occurred during initialization: {e}")
                self.simtime = -1

            if not _save:
                del mutator
                return None, None, None

            # creace the mutated_seed file
            trace_path = self.save_scenario_queue(_seed)

            # get rid of collision seed
            if mutator.is_collision:
                self.remove_scenario_data(_seed)
                print("Broken Seed!")
                del mutator
                return None, None, 1

            else:
                # get muation info
                mutated_seed, score, mutation_info = mutator.excuate_initial_mutation(desired_occlusion=self.desired_occlusion)
             #   mutator.clean()
                del mutator

                score = 1 / (abs(score - self.desired_occlusion) + 1)
                mutated_scenario = Scenario(trace_path)
                with open(trace_path, "w") as outfile:
                    json.dump(mutated_seed, outfile, indent=4)
                mutated_scenario.update_instance()

                return mutated_scenario, score, mutation_info

        else:
            # calculate feedback score

            if  _seed.instance["start_frame"] is not None:
                try:
                    if _seed.hash is None:
                        data_dir = f"{self.workdir}/{str(self.strategy)}/{str(self.desired_occlusion)}/dataset/{_seed.get_hash()}"
                    else:
                        data_dir = f"{self.workdir}/{str(self.strategy)}/{str(self.desired_occlusion)}/dataset/{_seed.hash}"

                    mutator.run_sim(_seed.instance, self.client, global_cnt=self.global_cnt, data_dir=data_dir,
                                    desired_occlusion=self.desired_occlusion)
                except Exception as e:
                    print(f"An error occurred during sim: {e}")
                    self.simtime = -1

                if not _save:
                    del mutator
                    return None, None, None

                # save scenario data
                trace_path = self.save_scenario_queue(_seed)
                #  get rid of collision seed
                if mutator.is_collision:
                    self.remove_scenario_data(_seed)
                    print("Broken Seed!")
                    del mutator
                    return None, None, 1
                else:
                    # get muation info
                    if self.global_cnt < 10:
                        preference = random.choice([0, 1, 2])
                        if preference == 1:
                            mutated_seed, score, mutation_info = mutator.excuate_adjust_mutation(
                                desired_occlusion=self.desired_occlusion)
                        elif preference == 2:
                            mutated_seed, score, mutation_info = mutator.excuate_adjust_triggering_time(desired_occlusion=self.desired_occlusion)
                        else:
                            mutated_seed, score, mutation_info = mutator.excuate_adjust_speed(desired_occlusion=self.desired_occlusion)

                        self.global_cnt += 1

                    else:
                        mutated_seed, score, mutation_info = mutator.excuate_spawn_mutation()
                        self.global_cnt = 0

                    # the more the mean score is closed to the desired_visbility the higher score the seed get
                    score = 1 / (abs(score - self.desired_occlusion) + 1)
                    #       mutator.clean()
                    del mutator

                mutated_scenario = Scenario(trace_path)

                with open(trace_path, "w") as outfile:
                    json.dump(mutated_seed, outfile, indent=4)

                mutated_scenario.update_instance()

            else:
                trace_path = self.save_scenario_queue(_seed)
                self.remove_scenario_data(_seed)
                print("Invalid Seed!")
                del mutator
                return None, None, 1


            return mutated_scenario, score, mutation_info

    def _run_instance(self, _seed, _save=True):
        """
        Run a single instance of the simulation with the given seed.
        Args:
            _seed: The seed to run
            _save: Whether to save the results
        Returns:
            tuple: (mutated_scenario, score, mutation_info)
        """
        self.restart_simulation()
        mutator = _Mutator.exp_Mutator(_seed.seed_path)
        try:
            mutator.run_sim(_seed.instance, self.client, global_cnt=self.global_cnt, data_dir=data_dir,
                            desired_occlusion=self.desired_occlusion)
        except Exception as e:
            print(f"An error occurred during simulation: {e}")
            self.simtime = -1

        if not _save:
            del mutator
            return None, None, None

        trace_path = self.save_scenario_queue(_seed)
        if mutator.is_collision:
            data_path_dir = os.path.join(self.tracking_path, mutator.hash_value)
            print("Broken Seed!")
            del mutator
            return None, None, 1
        else:
            if self.global_cnt < 10:
                preference = random.choice([0, 1, 2])
                if preference == 1:
                    mutated_seed, score, mutation_info = mutator.excuate_adjust_mutation(
                        desired_occlusion=self.desired_occlusion)
                elif preference == 2:
                    mutated_seed, score, mutation_info = mutator.excuate_adjust_triggering_time(desired_occlusion=self.desired_occlusion)
                else:
                    mutated_seed, score, mutation_info = mutator.excuate_adjust_speed(desired_occlusion=self.desired_occlusion)

                self.global_cnt += 1
            else:
                mutated_seed, score, mutation_info = mutator.excuate_spawn_mutation()
                self.global_cnt = 0

            score = 1 / (abs(score - self.desired_occlusion) + 1)
            self.tracking_path = os.path.join(self.tracking_path, mutator.hash_value)
            del mutator

        mutated_scenario = Scenario(trace_path)
        with open(trace_path, "w") as outfile:
            json.dump(mutated_seed, outfile, indent=4)
        mutated_scenario.update_instance()

        return mutated_scenario, score, mutation_info

    def population_add(self, mutated_seed):
        scenarios = mutated_seed
        # self.population.append(_seed)
        self.population.insert(0, scenarios)

        self.population.sort(key=lambda x: x.score, reverse=False)
        self.population = self.population[:20]

    def population_get(self):
        def softmax_with_temperature(data, temperature=1.0):
            scaled_data = data / temperature
            exp_data = np.exp(scaled_data - np.max(scaled_data))
            return exp_data / np.sum(exp_data)

        energy = list(map(lambda seed: seed.score, self.population))
        energy = np.array(energy)
        temperature = 1 / len(self.population)
        norm_energy = softmax_with_temperature(energy, temperature=temperature)
        idx = np.random.choice(len(self.population), p=norm_energy)
        seed_pool = [f"{self.population[i].score}({norm_energy[i]:.2f})" for i in range(len(self.population))]

        seed = self.population[idx]
        LOG.info("seed pool: {}".format(seed_pool))
        LOG.info("get seed score: {}".format(seed.score))
        return seed, idx

    def population_dup(self, score, cur_scenario):
        count_zero = 0
        for seed in self.population:
            if score != 0 and abs(seed.score - score) < 0.005 :
                return True
            if score == 0:
                count_zero += 1
        if count_zero > 20:
            return True

        return False

    def population_pop(self, scenario_to_remove):
        """
        Remove the specified scenario from the population.
        Args:
            scenario_to_remove: The scenario (tuple) that needs to be removed from the population.
        """
        # Check if the specified scenario exists in the population
        if scenario_to_remove in self.population:
            self.population.remove(scenario_to_remove)
            seed_path = scenario_to_remove.seed_path
            LOG.info("Removed the specified scenario from the population.")
            if os.path.exists(seed_path):
                try:
                    os.remove(seed_path)
                    LOG.info(f"Successfully deleted the file at {seed_path}.")
                except Exception as e:
                    LOG.error(f"Error while deleting the file at {seed_path}: {e}")
            else:
                LOG.warning(f"File at {seed_path} does not exist, skipping deletion.")

            LOG.info("Removed the specified scenario from the population.")
        else:
            LOG.info("The specified scenario is not in the population.")

    def execute(self):
        if self.flag < len(self.corpus):
            # check every scenario in the corpus
            scenario = self.corpus[self.flag]
            LOG.info(scenario.seed_path)
            print(scenario.seed_path)

            self.save_snapshot()
            _mutated_scenario, _score, _ = self._run_instance(scenario)


            print(f"seed:{scenario.seed_path}, occ_score:{_score}")
            # check if the scenario is the desired seed
            if abs(_score - 1) < 1 / 10:
                self.save_scenario_trace(scenario)

            else:
                if _mutated_scenario is not None:
                    scenario.verified = True
                    scenario.occ_score = _score
                    _mutated_scenario.score = _score
                    self.population_add(_mutated_scenario)
                    # remove unwanted data
                    self.flag += 1
                    self.save_snapshot()

                else:
                    if _ is None:
                        scenario.verified = False
                    elif _ is 1:
                        self.population_pop(scenario)

        else:
            if len(self.population) < 3:
                result = random.choice([True, False])
                if result:
                    scenario = random.choice(self.corpus)
                    LOG.info(scenario.seed_path)
                    print(scenario.seed_path)
                    self.save_snapshot()
                    _mutated_scenario, _score, _ = self._run_instance(scenario)

                    if abs(_score - 1) < 1 / 10:
                        self.save_scenario_trace(scenario)
                        scenario.verified = True

                    else:
                        if _mutated_scenario is not None:
                            scenario.verified = True
                            scenario.occ_score = _score
                            _mutated_scenario.score = _score
                            self.population_add(_mutated_scenario)
                            self.save_snapshot()
                        else:
                            if _ is None:
                                scenario.verified = False
                            elif _ is 1:
                                self.population_pop(scenario)


        # the fuzzing loop
        while len(self.population) > 0 :
            max_score = max(self.population, key=lambda x: x.score).score if self.population else 0.0
            probability = (1.0 - max_score) ** 2 + 0.25
            if random.random() < probability:
                scenario = random.choice(self.corpus)
                LOG.info(f"Selected scenario from corpus: {scenario.seed_path}")
                print(scenario.seed_path)

                self.save_snapshot()
                _mutated_scenario, _score, _ = self._run_instance(scenario)

                if _score is None:
                    LOG.warning("Skipping scenario due to invalid score.")
                    return

                if abs(_score - 1) < 1 / 10:
                    self.save_scenario_trace(scenario)
                    scenario.verified = True
                else:
                    if _mutated_scenario is not None:
                        scenario.verified = True
                        scenario.occ_score = _score
                        _mutated_scenario.score = _score
                        self.population_add(_mutated_scenario)
                        self.save_snapshot()
                    else:
                        if _ is None:
                            scenario.verified = False
                        elif _ == 1:
                            self.population_pop(scenario)

            else:
                cur_scenario, idx = self.population_get()
                print(cur_scenario.seed_path)
                self.save_snapshot()
                if cur_scenario.instance["cnt"] > 6:
                    self.population_pop(cur_scenario)
                    self.save_snapshot()
                re_mutated_scenario, re_score, mutation_info = self._run_instance(cur_scenario)
                print(f"seed:{cur_scenario.seed_path}, occ_score:{re_score}")
                if re_score is not None and abs(re_score - 1) < 1 / 10:
                    cur_scenario.instance["right_window"] = re_mutated_scenario.instance["right_window"]
                    cur_scenario.instance["left_window"] = re_mutated_scenario.instance["left_window"]
                    with open(cur_scenario.seed_path, "w") as outfile:
                        json.dump(cur_scenario.instance, outfile, indent=4)
                    self.save_scenario_trace(cur_scenario)
                    self.population_pop(cur_scenario)
                    self.save_snapshot()

                else:
                    if re_mutated_scenario is not None:
                        LOG.info("Original seed: {}".format(cur_scenario.seed_path))
                        LOG.info("Saved seed: {}".format(cur_scenario.hash))
                        LOG.info("Queue size: {}".format(len(self.population)))
                        cur_scenario.verified = True
                        cur_scenario.occ_score = re_score
                        if re_score < cur_scenario.score:
                            cur_scenario.score = re_score

                        cur_scenario.instance["right_window"] = re_mutated_scenario.instance["right_window"]
                        with open(cur_scenario.seed_path, "w") as outfile:
                            json.dump(cur_scenario.instance, outfile, indent=4)
                        self.population[idx] = cur_scenario
                        re_mutated_scenario.score = re_score

                        if self.population_dup(re_score, cur_scenario):
                            print("Similar seed: {}".format(cur_scenario.seed_path))

                        else:
                            # to run the mutated seed
                            self.population_add(re_mutated_scenario)
                            print("ADD POPULATION!")
                            self.hashes.add(cur_scenario.hash)

                        self.save_snapshot()

                        log_info = f"{cur_scenario.hash}\t{cur_scenario.score}\t{mutation_info}\t{re_mutated_scenario.seed_path}\n"
                        with open(self.workdir + f"/{str(self.strategy)}/{str(self.desired_occlusion)}/info", "a") as f:
                            f.write(log_info)

                    else:
                        if mutation_info is 1:
                            self.population_pop(cur_scenario)



        return 0

    def save_snapshot(self):
        snapshot_path = self.workdir + f"/{str(self.strategy)}/{str(self.desired_occlusion)}/snapshot"
        snapshot = {
            "simtime": self.simtime,
            "global_cnt": self.global_cnt,
            "flag": self.flag,
            "population": [
                {
                    "seed": {
                        "occ_score": seed.occ_score,
                        "score": seed.score,
                        "verified": seed.verified,
                        "seed_path": seed.seed_path,
                        "hash": seed.hash
                    }

                }
                for seed in self.population
            ],
            "hashes": list(self.hashes),
            "carla_process": "Running" if self.carla_process else "Stopped",
            "strategy": self.strategy,
            "desired_occlusion": self.desired_occlusion
        }

        with open(snapshot_path, "w") as f:
            json.dump(snapshot, f, indent=4)
        LOG.info(f"Snapshot saved at {snapshot_path}")


    def load_snapshot(self):
        snapshot_path = self.workdir + f"/{str(self.strategy)}/{str(self.desired_occlusion)}/snapshot"
        try:
            with open(snapshot_path, "r") as f:
                snapshot = json.load(f)

            self.simtime = snapshot.get("simtime", 0)
            self.hashes = set(snapshot.get("hashes", []))
            self.global_cnt = snapshot.get("global_cnt", 0)
            self.strategy = snapshot.get("strategy", "")
            self.desired_occlusion = snapshot.get("desired_occlusion", 0.8)
            self.flag = snapshot.get("flag", 0)

            # Append to the population list instead of replacing it
            for seed in snapshot["population"]:
                origin_seed_data = seed["seed"]
                origin_seed = Scenario(origin_seed_data["seed_path"])
                origin_seed.occ_score = origin_seed_data["occ_score"]
                origin_seed.score = origin_seed_data["score"]
                origin_seed.verified = origin_seed_data["verified"]
                origin_seed.hash = origin_seed_data["hash"]

                # Append the new origin_seed and mutated_seed to the population list
                self.population.append(origin_seed)

            LOG.info(f"Snapshot loaded from {snapshot_path}")

        except Exception as e:
            LOG.error(f"Error loading snapshot: {str(e)}")
            return False

        return True

    def excuate_random(self):
        if self.flag < len(self.corpus):
            # check every scenario in the corpus
            scenario = self.corpus[self.flag]
            LOG.info(scenario.seed_path)
            print(scenario.seed_path)

            self.save_snapshot()
            _mutated_scenario, _score, _ = self._run_instance(scenario)


            print(f"seed:{scenario.seed_path}, occ_score:{_score}")
            # check if the scenario is the desired seed
            if abs(_score - 1) < 1 / 10:
                self.save_scenario_trace(scenario)

            else:
                if _mutated_scenario is not None:
                    scenario.verified = True
                    scenario.occ_score = _score
                    _mutated_scenario.score = _score
                    self.population_add(_mutated_scenario)
                    # remove unwanted data
                    self.flag += 1
                    self.save_snapshot()

                else:
                    if _ is None:
                        scenario.verified = False
                    elif _ is 1:
                        self.population_pop(scenario)

        else:
            if len(self.population) < 3:
                result = random.choice([True, False])
                if result:
                    scenario = random.choice(self.corpus)
                    LOG.info(scenario.seed_path)
                    print(scenario.seed_path)
                    self.save_snapshot()
                    _mutated_scenario, _score, _ = self._run_instance(scenario)

                    if abs(_score - 1) < 1 / 10:
                        self.save_scenario_trace(scenario)
                        scenario.verified = True

                    else:
                        if _mutated_scenario is not None:
                            scenario.verified = True
                            scenario.occ_score = _score
                            _mutated_scenario.score = _score
                            self.population_add(_mutated_scenario)
                            self.save_snapshot()
                        else:
                            if _ is None:
                                scenario.verified = False
                            elif _ is 1:
                                self.population_pop(scenario)


        # the fuzzing loop
        while len(self.population) > 0 :
            max_score = max(self.population, key=lambda x: x.score).score if self.population else 0.0
            probability = (1.0 - max_score) ** 2 + 0.25
            if random.random() < probability:
                scenario = random.choice(self.corpus)
                LOG.info(f"Selected scenario from corpus: {scenario.seed_path}")
                print(scenario.seed_path)

                self.save_snapshot()
                _mutated_scenario, _score, _ = self._run_instance(scenario)

                if _score is None:
                    LOG.warning("Skipping scenario due to invalid score.")
                    return

                if abs(_score - 1) < 1 / 20:
                    self.save_scenario_trace(scenario)
                    scenario.verified = True
                else:
                    if _mutated_scenario is not None:
                        scenario.verified = True
                        scenario.occ_score = _score
                        _mutated_scenario.score = _score
                        self.population_add(_mutated_scenario)
                        self.save_snapshot()
                    else:
                        if _ is None:
                            scenario.verified = False
                        elif _ == 1:
                            self.population_pop(scenario)

            else:
                cur_scenario, idx = self.population_get()
                print(cur_scenario.seed_path)
                self.save_snapshot()
                if cur_scenario.instance["cnt"] > 6:
                    self.population_pop(cur_scenario)
                    self.save_snapshot()
                re_mutated_scenario, re_score, mutation_info = self._run_instance(cur_scenario)
                print(f"seed:{cur_scenario.seed_path}, occ_score:{re_score}")
                if re_score is not None and abs(re_score - 1) < 1 / 10:
                    cur_scenario.instance["right_window"] = re_mutated_scenario.instance["right_window"]
                    cur_scenario.instance["left_window"] = re_mutated_scenario.instance["left_window"]
                    with open(cur_scenario.seed_path, "w") as outfile:
                        json.dump(cur_scenario.instance, outfile, indent=4)
                    self.save_scenario_trace(cur_scenario)
                    self.population_pop(cur_scenario)
                    self.save_snapshot()

                else:
                    if re_mutated_scenario is not None:
                        LOG.info("Original seed: {}".format(cur_scenario.seed_path))
                        LOG.info("Saved seed: {}".format(cur_scenario.hash))
                        LOG.info("Queue size: {}".format(len(self.population)))
                        cur_scenario.verified = True
                        cur_scenario.occ_score = re_score
                        if re_score < cur_scenario.score:
                            cur_scenario.score = re_score

                        cur_scenario.instance["right_window"] = re_mutated_scenario.instance["right_window"]
                        with open(cur_scenario.seed_path, "w") as outfile:
                            json.dump(cur_scenario.instance, outfile, indent=4)
                        self.population[idx] = cur_scenario
                        re_mutated_scenario.score = re_score

                        if self.population_dup(re_score, cur_scenario):
                            print("Similar seed: {}".format(cur_scenario.seed_path))

                        else:
                            # to run the mutated seed
                            self.population_add(re_mutated_scenario)
                            print("ADD POPULATION!")
                            self.hashes.add(cur_scenario.hash)

                        self.save_snapshot()

                        log_info = f"{cur_scenario.hash}\t{cur_scenario.score}\t{mutation_info}\t{re_mutated_scenario.seed_path}\n"
                        with open(self.workdir + f"/{str(self.strategy)}/{str(self.desired_occlusion)}/info", "a") as f:
                            f.write(log_info)

                    else:
                        if mutation_info is 1:
                            self.population_pop(cur_scenario)

    def excuate_weak(self):
        seed_queue = corpus_parser(os.path.join(self.exp_path, str(self.strategy), str(self.desired_occlusion), "queue"))
        if len(seed_queue)>0:
            cur_scenario = random.choice(seed_queue)
            re_mutated_scenario, re_score, mutation_info = self.run_instance(cur_scenario)
            print(f"seed:{cur_scenario.seed_path}, occ_score:{re_score}")
            if re_score is not None and abs(re_score-1) < 1/10 :
                cur_scenario.instance["right_window"] = re_mutated_scenario.instance["right_window"]
                cur_scenario.instance["left_window"] = re_mutated_scenario.instance["left_window"]
                with open(cur_scenario.seed_path, "w") as outfile:
                    json.dump(cur_scenario.instance, outfile, indent=4)
                self.save_scenario_trace(cur_scenario)
            else:
                if re_mutated_scenario is not None:
                    cur_scenario.instance["right_window"] = re_mutated_scenario.instance["right_window"]
                    self.remove_scenario_data(cur_scenario)
        else:
            scenario = random.choice(self.corpus)
            _mutated_scenario, _score, _ = self.run_instance(scenario)
            self.remove_scenario_data(scenario)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Execute Simulation Fuzzing with configurable parameters.")

    parser.add_argument("--corpus", type=str, required=True, help="Path to the corpus directory.")
    parser.add_argument("--workdir", type=str, required=True, help="Path to the working directory.")
    parser.add_argument("--map", type=str, default="Town01", help="CARLA map to use.")
    parser.add_argument("--strategy", type=str, choices=["guided", "random", "no_scheduling"], required=True, help="Fuzzing strategy.")
    parser.add_argument("--desired_occlusion", type=float, required=True, help="Desired occlusion level.")
    parser.add_argument("--carla_path", type=str, default="./carla/CarlaUE4.sh", help="Path to CARLA executable.")
    parser.add_argument("--tracking_path", type=str, default="./data/tracking", help="Path to tracking data directory.")
    parser.add_argument("--exp_path", type=str, default="./data/exp", help="Path to experiment data directory.")
    parser.add_argument("--benchmark_path", type=str, default="./data/benchmark", help="Path to benchmark data directory.")

    return parser.parse_args()

def main():
    """Execute Simulation Fuzzing"""
    args = parse_arguments()

    print("Received arguments:", sys.argv)  # For debugging

    LOG.info("Fuzzing Start.")
    LOG.info(f"Strategy: {args.strategy}")
    LOG.info(f"Desired Occlusion: {args.desired_occlusion}")
    LOG.info(f"CARLA Path: {args.carla_path}")
    LOG.info(f"Tracking Path: {args.tracking_path}")
    LOG.info(f"Experiment Path: {args.exp_path}")
    LOG.info(f"Benchmark Path: {args.benchmark_path}")

    fuzzer = Fuzzer(
        _corpus=args.corpus,
        _workdir=args.workdir,
        _map=args.map,
        _strategy=args.strategy,
        _desired_occlusion=args.desired_occlusion,
        _carla_path=args.carla_path,
        _tracking_path=args.tracking_path,
        _exp_path=args.exp_path,
        _benchmark_path=args.benchmark_path
    )

    strategy_methods = {
        "guided": fuzzer.execute,
        "random": fuzzer.excuate_random,
        "no_scheduling": fuzzer.excuate_weak
    }

    strategy_method = strategy_methods.get(args.strategy)
    if strategy_method:
        strategy_method()
    else:
        LOG.error(f"Invalid strategy: {args.strategy}")
        exit(1)

    LOG.info("Fuzzing End.")

if __name__ == "__main__":
    main()
