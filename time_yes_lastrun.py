#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.2),
    on Tue Apr  9 08:26:00 2024
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from code
import random

card_width = 0.3
card_height = 0.5
position_y = 0.05
label_y = -0.25
condition = "sufficient_time"
positions = [(-0.6, position_y), (-0.2, position_y), (0.2, position_y), (0.6, position_y)]
# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.2'
expName = 'IOWA_Gambling_Task_incase'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/pushkarsingh/Documents/01 University/02 Psych of Emotions/03 Research Proposal/02 Experiment/001 Final Experiment Files/time_yes_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.EXP)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=[1440, 900], fullscr=False, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [-1.0000, -1.0000, -1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.mouseVisible = True
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='iohub')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='ioHub')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "defining_variables" ---
    # Run 'Begin Experiment' code from code
    total = 2000
    gamble_result_text = ""
    consequences = {
        "deck_A": {
            "reward": 100,
            "loss": 125,
            "loss_weight": 0.5,
        }, 
        "deck_B": {
            "reward": 100,
            "loss": 1250,
            "loss_weight": 0.1,
        }, 
        "deck_C": {
            "reward": 50,
            "loss": 25,
            "loss_weight": 0.5,
        }, 
        "deck_D": {
            "reward": 50,
            "loss": 250,
            "loss_weight": 0.1,
        },
    }
    selected_deck = ""
    random_val = 0
    
    # --- Initialize components for Routine "Instructions" ---
    text_norm = visual.TextStim(win=win, name='text_norm',
        text="In this game, you will be given four decks and 2000 dollars to gamble with. The deck you choose will get you some reward or some loss, based on the deck you choose you have to maximise the amount of money you make by the end of the game.\n\npress 'spacebar' to continue",
        font='Arial',
        units='norm', pos=(0, 0), height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instruct = keyboard.Keyboard()
    # Run 'Begin Experiment' code from text_align
    # Code components should usually appear at the top
    # of the routine. This one has to appear after the
    # text component it refers to.
    text_norm.alignText= 'left'
    
    # --- Initialize components for Routine "time_instructions" ---
    text_norm_3 = visual.TextStim(win=win, name='text_norm_3',
        text="Every time you get a choice, you will have 4 seconds to respond which is sufficient to learn and complete the task.\n\npress 'spacebar' to continue",
        font='Arial',
        units='norm', pos=(0, 0), height=0.1, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_instruct_3 = keyboard.Keyboard()
    # Run 'Begin Experiment' code from text_align_3
    # Code components should usually appear at the top
    # of the routine. This one has to appear after the
    # text component it refers to.
    text_norm.alignText= 'left'
    
    # --- Initialize components for Routine "countdown" ---
    text_countdown = visual.TextStim(win=win, name='text_countdown',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.06, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "Task" ---
    current_total = visual.TextStim(win=win, name='current_total',
        text='',
        font='Open Sans',
        pos=(0, 0.40), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    deck_A = visual.ImageStim(
        win=win,
        name='deck_A', 
        image='deck_image.png', mask=None, anchor='center',
        ori=0.0, pos=[positions[0]], size=(card_width, card_height),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    deck_B = visual.ImageStim(
        win=win,
        name='deck_B', 
        image='deck_image.png', mask=None, anchor='center',
        ori=0.0, pos=[positions[1]], size=(card_width, card_height),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    deck_C = visual.ImageStim(
        win=win,
        name='deck_C', 
        image='deck_image.png', mask=None, anchor='center',
        ori=0.0, pos=[positions[2]], size=(card_width, card_height),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    deck_D = visual.ImageStim(
        win=win,
        name='deck_D', 
        image='deck_image.png', mask=None, anchor='center',
        ori=0.0, pos=[positions[3]], size=(card_width, card_height),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-5.0)
    deck_A_label = visual.TextStim(win=win, name='deck_A_label',
        text='Deck A',
        font='Open Sans',
        pos=(positions[0][0], label_y), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    deck_B_label = visual.TextStim(win=win, name='deck_B_label',
        text='Deck B',
        font='Open Sans',
        pos=(positions[1][0], label_y), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    deck_C_label = visual.TextStim(win=win, name='deck_C_label',
        text='Deck C',
        font='Open Sans',
        pos=(positions[2][0], label_y), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    deck_D_label = visual.TextStim(win=win, name='deck_D_label',
        text='Deck D',
        font='Open Sans',
        pos=(positions[3][0], label_y), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-9.0);
    mouse = event.Mouse(win=win)
    x, y = [None, None]
    mouse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "gamble_result" ---
    text = visual.TextStim(win=win, name='text',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "defining_variables" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('defining_variables.started', globalClock.getTime())
    # keep track of which components have finished
    defining_variablesComponents = []
    for thisComponent in defining_variablesComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "defining_variables" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in defining_variablesComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "defining_variables" ---
    for thisComponent in defining_variablesComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('defining_variables.stopped', globalClock.getTime())
    # the Routine "defining_variables" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Instructions" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Instructions.started', globalClock.getTime())
    key_instruct.keys = []
    key_instruct.rt = []
    _key_instruct_allKeys = []
    # keep track of which components have finished
    InstructionsComponents = [text_norm, key_instruct]
    for thisComponent in InstructionsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Instructions" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_norm* updates
        
        # if text_norm is starting this frame...
        if text_norm.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_norm.frameNStart = frameN  # exact frame index
            text_norm.tStart = t  # local t and not account for scr refresh
            text_norm.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_norm, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_norm.status = STARTED
            text_norm.setAutoDraw(True)
        
        # if text_norm is active this frame...
        if text_norm.status == STARTED:
            # update params
            pass
        
        # *key_instruct* updates
        waitOnFlip = False
        
        # if key_instruct is starting this frame...
        if key_instruct.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_instruct.frameNStart = frameN  # exact frame index
            key_instruct.tStart = t  # local t and not account for scr refresh
            key_instruct.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_instruct, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_instruct.started')
            # update status
            key_instruct.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_instruct.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_instruct.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_instruct.status == STARTED and not waitOnFlip:
            theseKeys = key_instruct.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_instruct_allKeys.extend(theseKeys)
            if len(_key_instruct_allKeys):
                key_instruct.keys = _key_instruct_allKeys[0].name  # just the first key pressed
                key_instruct.rt = _key_instruct_allKeys[0].rt
                key_instruct.duration = _key_instruct_allKeys[0].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in InstructionsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Instructions" ---
    for thisComponent in InstructionsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Instructions.stopped', globalClock.getTime())
    # check responses
    if key_instruct.keys in ['', [], None]:  # No response was made
        key_instruct.keys = None
    thisExp.addData('key_instruct.keys',key_instruct.keys)
    if key_instruct.keys != None:  # we had a response
        thisExp.addData('key_instruct.rt', key_instruct.rt)
        thisExp.addData('key_instruct.duration', key_instruct.duration)
    thisExp.nextEntry()
    # the Routine "Instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "time_instructions" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('time_instructions.started', globalClock.getTime())
    key_instruct_3.keys = []
    key_instruct_3.rt = []
    _key_instruct_3_allKeys = []
    # keep track of which components have finished
    time_instructionsComponents = [text_norm_3, key_instruct_3]
    for thisComponent in time_instructionsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "time_instructions" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_norm_3* updates
        
        # if text_norm_3 is starting this frame...
        if text_norm_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_norm_3.frameNStart = frameN  # exact frame index
            text_norm_3.tStart = t  # local t and not account for scr refresh
            text_norm_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_norm_3, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_norm_3.status = STARTED
            text_norm_3.setAutoDraw(True)
        
        # if text_norm_3 is active this frame...
        if text_norm_3.status == STARTED:
            # update params
            pass
        
        # *key_instruct_3* updates
        waitOnFlip = False
        
        # if key_instruct_3 is starting this frame...
        if key_instruct_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_instruct_3.frameNStart = frameN  # exact frame index
            key_instruct_3.tStart = t  # local t and not account for scr refresh
            key_instruct_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_instruct_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_instruct_3.started')
            # update status
            key_instruct_3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_instruct_3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_instruct_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_instruct_3.status == STARTED and not waitOnFlip:
            theseKeys = key_instruct_3.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_instruct_3_allKeys.extend(theseKeys)
            if len(_key_instruct_3_allKeys):
                key_instruct_3.keys = _key_instruct_3_allKeys[0].name  # just the first key pressed
                key_instruct_3.rt = _key_instruct_3_allKeys[0].rt
                key_instruct_3.duration = _key_instruct_3_allKeys[0].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in time_instructionsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "time_instructions" ---
    for thisComponent in time_instructionsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('time_instructions.stopped', globalClock.getTime())
    # check responses
    if key_instruct_3.keys in ['', [], None]:  # No response was made
        key_instruct_3.keys = None
    thisExp.addData('key_instruct_3.keys',key_instruct_3.keys)
    if key_instruct_3.keys != None:  # we had a response
        thisExp.addData('key_instruct_3.rt', key_instruct_3.rt)
        thisExp.addData('key_instruct_3.duration', key_instruct_3.duration)
    thisExp.nextEntry()
    # the Routine "time_instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "countdown" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('countdown.started', globalClock.getTime())
    # keep track of which components have finished
    countdownComponents = [text_countdown]
    for thisComponent in countdownComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "countdown" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 10.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_countdown* updates
        
        # if text_countdown is starting this frame...
        if text_countdown.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_countdown.frameNStart = frameN  # exact frame index
            text_countdown.tStart = t  # local t and not account for scr refresh
            text_countdown.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_countdown, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_countdown.started')
            # update status
            text_countdown.status = STARTED
            text_countdown.setAutoDraw(True)
        
        # if text_countdown is active this frame...
        if text_countdown.status == STARTED:
            # update params
            text_countdown.setText("The task begins in" + "\n" + str(10-int(t)), log=False)
        
        # if text_countdown is stopping this frame...
        if text_countdown.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_countdown.tStartRefresh + 10-frameTolerance:
                # keep track of stop time/frame for later
                text_countdown.tStop = t  # not accounting for scr refresh
                text_countdown.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_countdown.stopped')
                # update status
                text_countdown.status = FINISHED
                text_countdown.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in countdownComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "countdown" ---
    for thisComponent in countdownComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('countdown.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-10.000000)
    
    # set up handler to look after randomisation of conditions etc
    trials_2 = data.TrialHandler(nReps=10.0, method='fullRandom', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='trials_2')
    thisExp.addLoop(trials_2)  # add the loop to the experiment
    thisTrial_2 = trials_2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
    if thisTrial_2 != None:
        for paramName in thisTrial_2:
            globals()[paramName] = thisTrial_2[paramName]
    
    for thisTrial_2 in trials_2:
        currentLoop = trials_2
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
        if thisTrial_2 != None:
            for paramName in thisTrial_2:
                globals()[paramName] = thisTrial_2[paramName]
        
        # set up handler to look after randomisation of conditions etc
        trials = data.TrialHandler(nReps=1.0, method='fullRandom', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions('gambling.xlsx'),
            seed=None, name='trials')
        thisExp.addLoop(trials)  # add the loop to the experiment
        thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        for thisTrial in trials:
            currentLoop = trials
            thisExp.timestampOnFlip(win, 'thisRow.t')
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    inputs=inputs, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
            if thisTrial != None:
                for paramName in thisTrial:
                    globals()[paramName] = thisTrial[paramName]
            
            # --- Prepare to start Routine "Task" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('Task.started', globalClock.getTime())
            # Run 'Begin Routine' code from task_code
            reward = ""
            loss = ""
            current_total.setText("Current Total : $" + str(total))
            # setup some python lists for storing info about the mouse
            mouse.x = []
            mouse.y = []
            mouse.leftButton = []
            mouse.midButton = []
            mouse.rightButton = []
            mouse.time = []
            mouse.clicked_name = []
            gotValidClick = False  # until a click is received
            # keep track of which components have finished
            TaskComponents = [current_total, deck_A, deck_B, deck_C, deck_D, deck_A_label, deck_B_label, deck_C_label, deck_D_label, mouse]
            for thisComponent in TaskComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "Task" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *current_total* updates
                
                # if current_total is starting this frame...
                if current_total.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    current_total.frameNStart = frameN  # exact frame index
                    current_total.tStart = t  # local t and not account for scr refresh
                    current_total.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(current_total, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'current_total.started')
                    # update status
                    current_total.status = STARTED
                    current_total.setAutoDraw(True)
                
                # if current_total is active this frame...
                if current_total.status == STARTED:
                    # update params
                    pass
                
                # *deck_A* updates
                
                # if deck_A is starting this frame...
                if deck_A.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    deck_A.frameNStart = frameN  # exact frame index
                    deck_A.tStart = t  # local t and not account for scr refresh
                    deck_A.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(deck_A, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'deck_A.started')
                    # update status
                    deck_A.status = STARTED
                    deck_A.setAutoDraw(True)
                
                # if deck_A is active this frame...
                if deck_A.status == STARTED:
                    # update params
                    pass
                
                # *deck_B* updates
                
                # if deck_B is starting this frame...
                if deck_B.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    deck_B.frameNStart = frameN  # exact frame index
                    deck_B.tStart = t  # local t and not account for scr refresh
                    deck_B.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(deck_B, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'deck_B.started')
                    # update status
                    deck_B.status = STARTED
                    deck_B.setAutoDraw(True)
                
                # if deck_B is active this frame...
                if deck_B.status == STARTED:
                    # update params
                    pass
                
                # *deck_C* updates
                
                # if deck_C is starting this frame...
                if deck_C.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    deck_C.frameNStart = frameN  # exact frame index
                    deck_C.tStart = t  # local t and not account for scr refresh
                    deck_C.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(deck_C, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'deck_C.started')
                    # update status
                    deck_C.status = STARTED
                    deck_C.setAutoDraw(True)
                
                # if deck_C is active this frame...
                if deck_C.status == STARTED:
                    # update params
                    pass
                
                # *deck_D* updates
                
                # if deck_D is starting this frame...
                if deck_D.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    deck_D.frameNStart = frameN  # exact frame index
                    deck_D.tStart = t  # local t and not account for scr refresh
                    deck_D.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(deck_D, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'deck_D.started')
                    # update status
                    deck_D.status = STARTED
                    deck_D.setAutoDraw(True)
                
                # if deck_D is active this frame...
                if deck_D.status == STARTED:
                    # update params
                    pass
                
                # *deck_A_label* updates
                
                # if deck_A_label is starting this frame...
                if deck_A_label.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    deck_A_label.frameNStart = frameN  # exact frame index
                    deck_A_label.tStart = t  # local t and not account for scr refresh
                    deck_A_label.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(deck_A_label, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'deck_A_label.started')
                    # update status
                    deck_A_label.status = STARTED
                    deck_A_label.setAutoDraw(True)
                
                # if deck_A_label is active this frame...
                if deck_A_label.status == STARTED:
                    # update params
                    pass
                
                # *deck_B_label* updates
                
                # if deck_B_label is starting this frame...
                if deck_B_label.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    deck_B_label.frameNStart = frameN  # exact frame index
                    deck_B_label.tStart = t  # local t and not account for scr refresh
                    deck_B_label.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(deck_B_label, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'deck_B_label.started')
                    # update status
                    deck_B_label.status = STARTED
                    deck_B_label.setAutoDraw(True)
                
                # if deck_B_label is active this frame...
                if deck_B_label.status == STARTED:
                    # update params
                    pass
                
                # *deck_C_label* updates
                
                # if deck_C_label is starting this frame...
                if deck_C_label.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    deck_C_label.frameNStart = frameN  # exact frame index
                    deck_C_label.tStart = t  # local t and not account for scr refresh
                    deck_C_label.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(deck_C_label, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'deck_C_label.started')
                    # update status
                    deck_C_label.status = STARTED
                    deck_C_label.setAutoDraw(True)
                
                # if deck_C_label is active this frame...
                if deck_C_label.status == STARTED:
                    # update params
                    pass
                
                # *deck_D_label* updates
                
                # if deck_D_label is starting this frame...
                if deck_D_label.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    deck_D_label.frameNStart = frameN  # exact frame index
                    deck_D_label.tStart = t  # local t and not account for scr refresh
                    deck_D_label.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(deck_D_label, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'deck_D_label.started')
                    # update status
                    deck_D_label.status = STARTED
                    deck_D_label.setAutoDraw(True)
                
                # if deck_D_label is active this frame...
                if deck_D_label.status == STARTED:
                    # update params
                    pass
                # *mouse* updates
                
                # if mouse is starting this frame...
                if mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    mouse.frameNStart = frameN  # exact frame index
                    mouse.tStart = t  # local t and not account for scr refresh
                    mouse.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(mouse, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.addData('mouse.started', t)
                    # update status
                    mouse.status = STARTED
                    mouse.mouseClock.reset()
                    prevButtonState = mouse.getPressed()  # if button is down already this ISN'T a new click
                if mouse.status == STARTED:  # only update if started and not finished!
                    buttons = mouse.getPressed()
                    if buttons != prevButtonState:  # button state changed?
                        prevButtonState = buttons
                        if sum(buttons) > 0:  # state changed to a new click
                            # check if the mouse was inside our 'clickable' objects
                            gotValidClick = False
                            clickableList = environmenttools.getFromNames([deck_A, deck_B, deck_C, deck_D], namespace=locals())
                            for obj in clickableList:
                                # is this object clicked on?
                                if obj.contains(mouse):
                                    gotValidClick = True
                                    mouse.clicked_name.append(obj.name)
                            if gotValidClick:
                                x, y = mouse.getPos()
                                mouse.x.append(x)
                                mouse.y.append(y)
                                buttons = mouse.getPressed()
                                mouse.leftButton.append(buttons[0])
                                mouse.midButton.append(buttons[1])
                                mouse.rightButton.append(buttons[2])
                                mouse.time.append(mouse.mouseClock.getTime())
                            if gotValidClick:
                                continueRoutine = False  # end routine on response
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in TaskComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Task" ---
            for thisComponent in TaskComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('Task.stopped', globalClock.getTime())
            # Run 'End Routine' code from task_code
            selected_deck = mouse.clicked_name[0]
            
            thisExp.addData('condition', condition)
            if selected_deck == "deck_A":
                reward = 100
                loss = deck_A_loss
                if loss == 0:
                    total = total + reward
                    gamble_result_text = f"You won ${reward}"
                else:
                    total = total + reward
                    total = total - loss
                    gamble_result_text = f"You won ${reward} and lost ${loss}"
            elif selected_deck == "deck_B":
                reward = 100
                loss = deck_B_loss
                if loss == 0:
                    total = total + reward
                    gamble_result_text = f"You won ${reward}"
                else:
                    total = total + reward
                    total = total - loss
                    gamble_result_text = f"You won ${reward} and lost ${loss}"
            elif selected_deck == "deck_C":
                reward = 50
                loss = deck_C_loss
                if loss == 0:
                    total = total + reward
                    gamble_result_text = f"You won ${reward}"
                else:
                    total = total + reward
                    total = total - loss
                    gamble_result_text = f"You won ${reward} and lost ${loss}"
            elif selected_deck == "deck_D":
                reward = 50
                loss = deck_D_loss
                if loss == 0:
                    total = total + reward
                    gamble_result_text = f"You won ${reward}"
                else:
                    total = total + reward
                    total = total - loss
                    gamble_result_text = f"You won ${reward} and lost ${loss}"
            else:
                gamble_result_text = selected_deck
                
            thisExp.addData("reward", reward)
            thisExp.addData("loss", loss)
            thisExp.addData("total", total)
            # store data for trials (TrialHandler)
            trials.addData('mouse.x', mouse.x)
            trials.addData('mouse.y', mouse.y)
            trials.addData('mouse.leftButton', mouse.leftButton)
            trials.addData('mouse.midButton', mouse.midButton)
            trials.addData('mouse.rightButton', mouse.rightButton)
            trials.addData('mouse.time', mouse.time)
            trials.addData('mouse.clicked_name', mouse.clicked_name)
            # the Routine "Task" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "gamble_result" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('gamble_result.started', globalClock.getTime())
            text.setText(gamble_result_text)
            # keep track of which components have finished
            gamble_resultComponents = [text]
            for thisComponent in gamble_resultComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "gamble_result" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 3.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text* updates
                
                # if text is starting this frame...
                if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text.frameNStart = frameN  # exact frame index
                    text.tStart = t  # local t and not account for scr refresh
                    text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text.started')
                    # update status
                    text.status = STARTED
                    text.setAutoDraw(True)
                
                # if text is active this frame...
                if text.status == STARTED:
                    # update params
                    pass
                
                # if text is stopping this frame...
                if text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text.tStartRefresh + 3.0-frameTolerance:
                        # keep track of stop time/frame for later
                        text.tStop = t  # not accounting for scr refresh
                        text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text.stopped')
                        # update status
                        text.status = FINISHED
                        text.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in gamble_resultComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "gamble_result" ---
            for thisComponent in gamble_resultComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('gamble_result.stopped', globalClock.getTime())
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-3.000000)
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'trials'
        
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 10.0 repeats of 'trials_2'
    
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
