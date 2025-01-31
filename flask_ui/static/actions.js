// actions.js

// Action Types
export const ACTION_TYPES = {
    LOAD_JOBS: 'LOAD_JOBS',
    SELECT_JOB: 'SELECT_JOB',
    UPDATE_RESULTS: 'UPDATE_RESULTS',
    SELECT_RESULT: 'SELECT_RESULT',
    ASSIGN_RESULT: 'ASSIGN_RESULT',
    CLEAR_ALL: 'CLEAR_ALL',
    EDIT_JOB_DESCRIPTION: 'EDIT_JOB_DESCRIPTION',
    UPDATE_ONE_RESULT: 'UPDATE_ONE_RESULT',
};

export const uncodableResult = {
    label: "*",
    description: "uncodable",
    distance: 9.99,
    rank: 9999,
}
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
    };
};

/**
 * Creates an action to clear all
 * @returns {Object} Action object
 */
export const clearAll = () => {
    localStorage.clear();
    return {
        type: ACTION_TYPES.CLEAR_ALL
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
    payload: { jobId, result }
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
});
