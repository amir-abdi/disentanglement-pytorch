"""
Borrowed from https://github.com/AIcrowd/neurips2019_disentanglement_challenge_starter_kit/blob/master/aicrowd_helpers.py
"""
#!/usr/bin/env python
import crowdai_api
import os

########################################################################
# Instatiate Event Notifier
########################################################################
crowdai_events = crowdai_api.events.CrowdAIEvents()


def execution_start():
    ########################################################################
    # Register Evaluation Start event
    ########################################################################
    print("Training Start...")
    crowdai_events.register_event(
                event_type=crowdai_events.CROWDAI_EVENT_INFO,
                message="execution_started",
                payload={ #Arbitrary Payload
                    "event_type": "disentanglement_challenge:execution_started"
                    }
                )


def register_progress(progress):
    ########################################################################
    # Register Evaluation Progress event
    # progress : float [0, 1]
    ########################################################################
    # print("Training Progress : {}".format(progress))
    crowdai_events.register_event(
                event_type=crowdai_events.CROWDAI_EVENT_INFO,
                message="register_progress",
                payload={ #Arbitrary Payload
                    "event_type": "disentanglement_challenge:register_progress",
                    "training_progress" : progress
                    }
                )

def submit(payload={}):
    ########################################################################
    # Register Evaluation Complete event
    ########################################################################
    print("AIcrowd Submit")
    crowdai_events.register_event(
                event_type=crowdai_events.CROWDAI_EVENT_SUCCESS,
                message="submit",
                payload={ #Arbitrary Payload
                    "event_type": "disentanglement_challenge:submit",
                    },
                blocking=True
                )

def execution_error(error):
    ########################################################################
    # Register Evaluation Complete event
    ########################################################################
    crowdai_events.register_event(
                event_type=crowdai_events.CROWDAI_EVENT_ERROR,
                message="execution_error",
                payload={ #Arbitrary Payload
                    "event_type": "disentanglement_challenge:execution_error",
                    "error" : error
                    },
                blocking=True
                )
