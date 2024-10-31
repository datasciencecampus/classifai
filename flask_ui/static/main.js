/**
 * @file main.js
 * @description Main entry point for the SIC/SOC Coding Tool application.
 */
// main.js
import { reducer, createStore } from './state.js';
import { ACTION_TYPES, loadJobs, updateResults, selectJob, selectResult, assignResult } from './actions.js';
import { loadSavedData, handleFileSelect, fetchResults } from './dataService.js';
import { initTables, populateJobTable, updateResultsTable, showJobDetails } from './uiService.js';

// Create the store
const store = createStore(reducer);

document.addEventListener('DOMContentLoaded', function() {

    // Initialise page objects
    const { jobTable, resultsDataTable } = initTables();
    // Re-load data on page refresh
    const { jobs, resultsData, selectedJobId, result } = loadSavedData();

    // Subscribe to store changes
    const unsubscribeLoadJobs = store.subscribe(ACTION_TYPES.LOAD_JOBS, (state,action) => {
        console.log('Update job table on:', action.type)
        populateJobTable(state.jobs, jobTable);
    });
    const unsubscribeSelectJob = store.subscribe(ACTION_TYPES.SELECT_JOB, (state,action) => {
        console.log('Change selected job on:', action.type)
        updateResultsTable(state.selectedJobId,state.resultsData,resultsDataTable);
        const job = state.jobs.find(job => job.id === state.selectedJobId);
        showJobDetails(job);
    });
    const unsubscribeUpdateResults = store.subscribe(ACTION_TYPES.UPDATE_RESULTS, (state,action) => {
        console.log('Update results table on:', action.type)
        updateResultsTable(state.selectedJobId,state.resultsData,resultsDataTable);
    });
    const unsubscribeAssignResult = store.subscribe(ACTION_TYPES.ASSIGN_RESULT, (state, action) => {
        console.log('Assigning code:', action.type);
        localStorage.setItem('jobsData',JSON.stringify(state.jobs));
        populateJobTable(state.jobs, jobTable);
        jobTable.row(state.selectedJobId).select();
    });

    // Update state (and listeners) with reloaded data after refresh
    store.dispatch(loadJobs(jobs));
    store.dispatch(updateResults(resultsData));
    if (selectedJobId !== null) {
        store.dispatch(selectJob(selectedJobId));
    }
    if (result && result !== null) {
        store.dispatch(selectResult(result));
    }

    /*
     * EVENT HANDLING
    */

    // Get page elements for events
    const fileInput = document.getElementById('csv-file');
    const searchButton = document.getElementById('fetch-results');
    const jobTableBody = document.getElementById('job-table').querySelector('tbody');
    const resultsTableBody = document.getElementById('results-table').querySelector('tbody');
    const assignResultButton = document.getElementById('assign-result');

    // File chooser event
    fileInput.addEventListener('change', async (event) => {
        const newJobs = await handleFileSelect(event);
        store.dispatch(loadJobs(newJobs));
    });

    // Click search button event
    searchButton.addEventListener('click', async () => {
        resultsDataTable.clear().draw();
        resultsDataTable.row.add({label: 'Loading...', description: 'Please wait', distance: ''}).draw();
        setTimeout(async () => {
            const newResultsData = await fetchResults(store.getState().jobs);
            store.dispatch(updateResults(newResultsData));
        }, 1000);
    });

    // Select job table row event
    jobTableBody.addEventListener('click', (event) => {
        const row = event.target.closest('tr');
        if (row) {
            const rowData = jobTable.row(row).data();
            store.dispatch(selectJob(rowData.id));
        }
    });

    // Select results table row event
    resultsTableBody.addEventListener('click', (event) => {
        const row = event.target.closest('tr');
        if (row) {
            const rowData = resultsDataTable.row(row).data() || {};
            if (rowData) {
                console.log('Selected code:',rowData.label)
                store.dispatch(selectResult(rowData));
            }
        }
    });

    // Assign result event
    assignResultButton.addEventListener('click', () => {
        const state = store.getState();
        if (state.selectedJobId && state.selectedResult.label) {
            store.dispatch(assignResult(state.selectedJobId, state.selectedResult));
        }
    });
});
