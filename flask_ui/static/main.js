/**
 * @file main.js
 * @description Main entry point for the SIC/SOC Coding Tool application.
 */
// main.js
import { store } from './state.js';
import { ACTION_TYPES, loadJobs, clearAll, newSession, updateResults, selectJob, selectResult, assignResult, editJobDescription, uncodableResult, updateOneResult, toggleCodedRows } from './actions.js';
import { upsertRecords, constructMockResponse, fetchResults, autocode, handleFixedWidthFileSelect, downloadFixedWidthFile, postJobsData, postResultsData } from './dataService.js';
import { initTables, populateJobTable, updateResultsTable, showJobDetails, incrementDataTable} from './uiService.js';

document.addEventListener('DOMContentLoaded', function() {

    // Initialise page objects
    const { jobTable, resultsDataTable } = initTables(store);
    // To access the app state
    // store.getState().{jobs|resultsData|selectedJobId|selectedResult}

    const unsubscribeToggleCodedRows = store.subscribe(ACTION_TYPES.TOGGLE_CODED_ROWS, (state, action) => {
        console.log('Toggling coded rows');
        populateJobTable(state.jobs, jobTable, state.hideCoded)
    });
    const unsubscribeNewSession = store.subscribe(ACTION_TYPES.NEW_SESSION, (state, action) => {
        console.log('New Session: ', state.sessionID)
        postJobsData(state.sessionID, state.jobs);
    });
    // Subscribe to store changes
    const unsubscribeLoadJobs = store.subscribe(ACTION_TYPES.LOAD_JOBS, (state,action) => {
        console.log('Update job table');
        populateJobTable(state.jobs, jobTable, state.hideCoded);
    });
    const unsubscribeSelectJob = store.subscribe(ACTION_TYPES.SELECT_JOB, (state,action) => {
        updateResultsTable(state.selectedJobId,state.resultsData,resultsDataTable);
        const job = state.jobs.find(job => job.id === state.selectedJobId);
        showJobDetails(job, (updatedJob) => { // State update callback for when user clicks save
            store.dispatch(editJobDescription(updatedJob));
        });
    });
    const unsubscribeUpdateResults = store.subscribe(ACTION_TYPES.UPDATE_RESULTS, (state,action) => {
        console.log('Update results table');
        updateResultsTable(state.selectedJobId,state.resultsData,resultsDataTable);
    });
    const unsubscribeAssignResult = store.subscribe(ACTION_TYPES.ASSIGN_RESULT, (state, action) => {
        console.log('Assigning code:', action.type);
        localStorage.setItem('jobsData',JSON.stringify(state.jobs));
        populateJobTable(state.jobs, jobTable, state.hideCoded);
        jobTable.row((idx,data) => data.id === state.selectedJobId).select();
        // jobTable.row('.selected').select();
        jobTable.table().node().focus();
    });
    const unsubscribeEditJobDescription = store.subscribe(ACTION_TYPES.EDIT_JOB_DESCRIPTION, (state,action) =>{
        console.log('Edit job details.')
        localStorage.setItem('jobsData',JSON.stringify(state.jobs));
        populateJobTable(state.jobs, jobTable, state.hideCoded);
        jobTable.row((idx,data) => data.id === state.selectedJobId).select();
        // jobTable.row('.selected').select();
        jobTable.table().node().focus();
    });
    const unsubscribeClearAll = store.subscribe(ACTION_TYPES.CLEAR_ALL, (state,action) =>{
        console.log('Clearing all and reloading the page.');
        window.location.reload();
    });
    const unsubscribeUpdateOneResult = store.subscribe(ACTION_TYPES.UPDATE_ONE_RESULT, (state, action) =>{
        localStorage.setItem('resultsData', JSON.stringify(state.resultsData));
        updateResultsTable(state.selectedJobId, state.resultsData, resultsDataTable);
    });

    // Update state (and listeners) with reloaded data after refresh
    store.dispatch(loadJobs(store.getState().jobs));
    store.dispatch(updateResults(store.getState().resultsData));
    if (store.getState().selectedJobId !== null) {
        store.dispatch(selectJob(store.getState().selectedJobId));
    }
    if (store.getState().selectedResult && store.getState().selectedResult !== null) {
        store.dispatch(selectResult(store.getState().selectedResult));
    }

    /*
     * EVENT HANDLING
    */

    // Get page elements for events
    const fileInput = document.getElementById('input-file');
    const clearButton = document.getElementById('clear-all');
    const downloadButton = document.getElementById('downloadButton');
    const searchButton = document.getElementById('fetch-results');
    const assignResultButton = document.getElementById('assign-result');
    const assignUncodableButton = document.getElementById('assign-uncodable');
    const toggleCodedRowsButton = document.getElementById('toggle-coded-rows');

    // Autocode event
    const autocodeMaxDistance = document.getElementById('autocode-max-distance');
    const autocodeMinDiff = document.getElementById('autocode-min-diff');
    const autocodeButton = document.getElementById('autocode-btn');
    autocodeButton.addEventListener('click', async () => {
        autocodeButton.disabled = true;
        try {
            const maxDistance = parseFloat(autocodeMaxDistance.value);
            const minDiff = parseFloat(autocodeMinDiff.value);
            console.log('Autocoding with',maxDistance,'and',minDiff);
            const newJobs = await Promise.resolve(
                autocode(store.getState().jobs, store.getState().resultsData, maxDistance, minDiff)
            );
            console.log('New jobs data length',newJobs.length);
            store.dispatch(loadJobs(newJobs));
        } finally {
            autocodeButton.disabled = false;
        }
    });

    // File input event
    fileInput.addEventListener('change', async (event) => {
        const newJobs = await handleFixedWidthFileSelect(event);
        newJobs.sort((a, b) => a.description.localeCompare(b.description));
        localStorage.setItem('jobsData', JSON.stringify(newJobs));

        store.dispatch(loadJobs(newJobs));
        setTimeout(() => {fileInput.value = ''},2000);

        // Search immediately on file load
        const loadingMessageResultsData = constructMockResponse(newJobs);
        store.dispatch(updateResults(loadingMessageResultsData));

        const updateCallback = (responseData,index) => {
            let currentResultsData = store.getState().resultsData;
            let newResultsData = upsertRecords(currentResultsData,responseData,record => record?.input_id);
            store.dispatch(updateResults(newResultsData));
        };
        // Create a new session on file input, then request from the api
        store.dispatch(newSession())
        fetchResults(newJobs,updateCallback, store.getState().sessionID, '/predict_soc',20,5);
    });

    // Backup in case file chooser 'change' event doesn't fire
    fileInput.addEventListener('click', () => {
        setTimeout(() => {fileInput.value = ''},2000);
    });

    // Download event (doesn't affect state)
    downloadButton.addEventListener('click', () => {
        downloadFixedWidthFile(store.getState().jobs, "results.txt");
        //downloadCSV(store.getState().jobs,"results.csv");
    });

    // Clear All event
    clearButton.addEventListener('click', () =>{
        if (confirm('Are you sure you want to clear all data? This action cannot be undone.')) {
            store.dispatch(clearAll());
        }
    })

    // Click search button event
    searchButton.addEventListener('click', async () => {
        const loadingMessageResultsData = constructMockResponse(store.getState().jobs);
        store.dispatch(updateResults(loadingMessageResultsData));
        const updateCallback = (responseData,index) => {
            let currentResultsData = store.getState().resultsData;
            let newResultsData = upsertRecords(currentResultsData,responseData,record => record?.input_id);
            store.dispatch(updateResults(newResultsData));
        };
        fetchResults(store.getState().jobs,
            updateCallback,
            '/predict_soc',
            20,
            5,
            );
    });

    // Assign result event
    assignResultButton.addEventListener('click', () => {
        const state = store.getState();
        if (state.selectedJobId && state.selectedResult.label) {
                const jobID = state.selectedJobId;
                const result = state.selectedResult;
                incrementDataTable(jobTable, (rowData) => store.dispatch(selectJob(rowData.id)));
                store.dispatch(assignResult(jobID, result));
            }
    });

    // Assign uncodable event
    assignUncodableButton.addEventListener('click', () => {
        const selectedJobId = store.getState().selectedJobId;
        if (selectedJobId) {
            incrementDataTable(jobTable, (rowData) => store.dispatch(selectJob(rowData.id)));
            store.dispatch(assignResult(selectedJobId, uncodableResult));
        }
    });

    // Click 'Get 1 result' button event
    const updateOneResultButton = document.getElementById('fetch-one-result');
    updateOneResultButton.addEventListener('click', async () => {
        const currentJob = store.getState().jobs.find(job => job.id === store.getState().selectedJobId);
        const loadingMessageCurrentResult = constructMockResponse([currentJob]);
        store.dispatch(updateOneResult(loadingMessageCurrentResult));
        const updateCallback = (responseData,index) => {
            let [newOneResult] = responseData;
            store.dispatch(updateOneResult(newOneResult));
        };
        fetchResults([currentJob],updateCallback);
    });

    // Toggle Coded Rows button event
    toggleCodedRowsButton.addEventListener('click', () => {
        const state = store.getState();
        store.dispatch(toggleCodedRows(state.hideCoded));
    });

});
