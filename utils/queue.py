from .jobs import Job
from .constants import *
from .helpers import *
from .data_io import *
from .sheet import read_sheet_info, write_sheet_info
from .operate import kill_jobs_tpu, check_tpu_status
from .users import user_from_dict

import .users

class Task:
    def __init__(
        self,
        job: Job,
        user,
        tpu_info,
        priority_info=None,
        job_info=None,
        other_info=None
    ):
    """
    job: the dict of the job
    tpu_info: a dict, consist of keys:
        - valid tpus: a list of tpu types
    priority_info: a dict, consist of keys:
    job_info: a dict, consist of keys:
        - stage_dir
    other_info: a dict, consist of keys:
        - task_id: a random number to identify the task
        - queue_time: the time when the task is in the queue

    """
        self.job = job
        self.user = user
        self.tpu_info = tpu_info
        self.priority_info = priority_info if priority_info is not None else {}
        self.job_info = job_info if job_info is not None else {}
        self.other_info = other_info if other_info is not None else {}

    def to_dict(self):
        return {
            "job": self.job.to_dict() if self.job else None,
            "user": self.user,
            "tpu_info": self.tpu_info,
            "priority_info": self.priority_info,
            "job_info": self.job_info,
            "other_info": self.other_info
        }

    @classmethod
    def from_dict(cls, data: dict):
        job_data = data.get("job")
        job_obj = Job.from_dict(job_data) if job_data else None
        return cls(
            job=job_obj,
            user=data.get("user"),
            tpu_info=data.get("tpu_info"),
            priority_info=data.get("priority_info", {}),
            job_info=data.get("job_info", {}),
            other_info=data.get("other_info", {})
        )

def run_task_on_tpu(task: Task, tpu):
    data = read_and_lock_data()
    try:
        user = data['users'][task.user]
        user_obj = user_from_dict(user)

def ack_queue(ack_information):
    """
    ack_information: a dict, consist of keys
        - tpu: the current tpu to be acknowledged
        - status: 'finished'/'failed'
    """
    pass


def update_staging_info(task_id, stage_dir, stage_time):
    queue = read_and_lock_queue()
    for task_dict in queue:
        if int(task_dict.get('other_info',{}).get('id',0)) == int(task_id):
            task_dict["other_info"]["queue_time"] = stage_time
            task_dict["job_info"]["stage_dir"] = stage_dir
            task_dict["job"]["stage_dir"] = stage_dir
    write_and_unlock_queue(queue)

def check_valid(task, information):
    """
    input:
        task: a task in the queue
        information: the empty TPU/other information of the current status

    output:
        True/False, indicate whether this task is valid
    """
    if (not "stage_dir" in task.job_info) or (not task.job_info["stage_dir"]): 
        return False

    