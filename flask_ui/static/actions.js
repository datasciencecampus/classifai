// actions.js

// Action Types
export const ACTION_TYPES = {
    NEW_SESSION: 'NEW_SESSION',
    LOAD_JOBS: 'LOAD_JOBS',
    SELECT_JOB: 'SELECT_JOB',
    UPDATE_RESULTS: 'UPDATE_RESULTS',
    SELECT_RESULT: 'SELECT_RESULT',
    ASSIGN_RESULT: 'ASSIGN_RESULT',
    CLEAR_ALL: 'CLEAR_ALL',
    EDIT_JOB_DESCRIPTION: 'EDIT_JOB_DESCRIPTION',
    UPDATE_ONE_RESULT: 'UPDATE_ONE_RESULT',
    TOGGLE_CODED_ROWS: 'TOGGLE_CODED_ROWS',
    LOAD_SESSION: 'LOAD_SESSION',
};

/**
 * The initial state of the application
 */
export const initialState = {
    jobs: [],
    selectedJobId: null,
    resultsData: [],
    selectedResult: {},
    hideCoded: false,
    sessionID: null,
  };


// Action Creators

/**
 * Creates an action to load jobs
 * @param {Array} jobs - Array of job objects
 * @returns {Object} Action object
 */
export const loadJobs = (jobs) => {
    localStorage.removeItem('selectedJobId');
    localStorage.setItem('jobsData', JSON.stringify(jobs));
    return {
        type: ACTION_TYPES.LOAD_JOBS,
        payload: jobs,
        stateUpdater: (state) => ({ ...state, jobs: jobs })
        }
    };


export const newSession = () => {
    const sessionID = uuid.v4();
    localStorage.setItem('sessionID', JSON.stringify(sessionID));
    return {
        type: ACTION_TYPES.NEW_SESSION,
        payload: sessionID,
        stateUpdater: (state) => ({ ...state, sessionID: sessionID })
    };
};

/**
 * Creates an action to clear all
 * @returns {Object} Action object
 */
export const clearAll = () => {
    localStorage.clear();
    return {
        type: ACTION_TYPES.CLEAR_ALL,
        stateUpdater: (state) => (initialState),
    };
};

/**
 * Creates an action to select a job
 * @param {number} jobId - ID of the selected job
 * @returns {Object} Action object
 */
export const selectJob = (jobId) => {
    localStorage.setItem('selectedJobId', JSON.stringify(jobId));
    return {
        type: ACTION_TYPES.SELECT_JOB,
        payload: jobId,
        stateUpdater: (state) => ({ ...state, selectedJobId: jobId }),
    };
};

/**
 * Creates an action to select a job
 * @param {number} jobId - ID of the selected job
 * @returns {Object} Action object
 */
 export const selectResult = (selectedResult) => {
    localStorage.setItem('selectedResult', JSON.stringify(selectedResult));
    console.log('Selected code:',selectedResult.label);
    return {
        type: ACTION_TYPES.SELECT_RESULT,
        payload: selectedResult,
        stateUpdater: (state) => ({ ...state, selectedResult: selectedResult }),
    };
};

/**
 * Creates an action to update results data
 * @param {Object} resultsData - Object containing results data
 * @returns {Object} Action object
 */
export const updateResults = (resultsData) => {
    localStorage.setItem('resultsData', JSON.stringify(resultsData));
    return {
        type: ACTION_TYPES.UPDATE_RESULTS,
        payload: resultsData,
        stateUpdater: (state) => ({ ...state, resultsData: resultsData }),
    };
};



/**
 * Creates an action to assign the result to the jobsData
 * @param {int} jobId - The id of the job row being written to
 * @param {Object} result - The result
 * @returns
 */
export const assignResult = (jobId, result) => ({
    type: ACTION_TYPES.ASSIGN_RESULT,
    payload: { jobId, result },
    stateUpdater: (state) => ({
        ...state,
        jobs: state.jobs.map(job =>
            job.id === jobId
                ? { ...job,
                    code: result.label,
                    code_description: result.description,
                    code_score: result.distance,
                    code_rank: result.rank,
                }
                : job
        )
    }),
});

/**
 * Creates an action to update the results list for one record
 *
 * @param {Object} resultSet - the result from the API call with slots: input_id (str) and response (Array)
 * @returns {Object} Action object
 */
export const updateOneResult = (resultSet) => {
    return {
        type: ACTION_TYPES.UPDATE_ONE_RESULT,
        payload: resultSet,
        stateUpdater: (state) => ({
            ...state,
            resultsData: upsertRecords(
              state.resultsData,
              resultSet,
              result => result?.input_id,
            )
          }),
    };
};

/**
 * Creates an action to assign the edited job description to the jobsData
 * @param {object} job - the new job record
 * @returns
 */
export const editJobDescription = (job) => ({
    type: ACTION_TYPES.EDIT_JOB_DESCRIPTION,
    payload: job,
    stateUpdater: (state) => ({
        ...state,
        jobs: upsertRecords(
          state.jobs,
          job,
        )
        }),
});

export const toggleCodedRows = (hiddenFlag) => {
    if (hiddenFlag == true) {
        var toggleOption = false;
    } else {
        var toggleOption = true;
    };
    localStorage.setItem('hideCoded', JSON.stringify(toggleOption));
    return {
        type: ACTION_TYPES.TOGGLE_CODED_ROWS,
        payload: toggleOption,
        stateUpdater: (state) => ({
            ...state,
            hideCoded: toggleOption,
            }),
    };
};

export const loadSession = (sessionID, jobsData, resultsData) => {
    localStorage.setItem('sessionID', JSON.stringify(sessionID));
    localStorage.setItem('jobsData', JSON.stringify(jobsData));
    localStorage.setItem('resultsData', JSON.stringify(resultsData));
    return {
        type: ACTION_TYPES.LOAD_SESSION,
        payload: [sessionID, jobsData, resultsData],
        stateUpdater: (state) => ({
            ...state,
            sessionID: sessionID,
            jobs: jobsData,
            resultsData: resultsData
        })
    };
};
