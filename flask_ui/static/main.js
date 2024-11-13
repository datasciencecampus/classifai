/**
 * @file main.js
 * @description Main entry point for the SIC/SOC Coding Tool application.
 */
// main.js
import { reducer, createStore } from './state.js';
import { ACTION_TYPES, loadJobs, updateResults, clearAll, selectJob, selectResult, assignResult, editJobDescription } from './actions.js';
import { loadSavedData, handleFileSelect, fetchResults, downloadCSV } from './dataService.js';
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
        showJobDetails(job, (updatedJob) => { // State update function for when user clicks save
            store.dispatch(editJobDescription(updatedJob));
        });
    });
    const unsubscribeUpdateResults = store.subscribe(ACTION_TYPES.UPDATE_RESULTS, (state,action) => {
        console.log('Update results table on:', action.type)
        updateResultsTable(state.selectedJobId,state.resultsData,resultsDataTable);
    });
    const unsubscribeAssignResult = store.subscribe(ACTION_TYPES.ASSIGN_RESULT, (state, action) => {
        console.log('Assigning code:', action.type);
        localStorage.setItem('jobsData',JSON.stringify(state.jobs));
        populateJobTable(state.jobs, jobTable);
        //jobTable.row(state.selectedJobId).select();
    });
    const unsubscribeClearAll = store.subscribe(ACTION_TYPES.CLEAR_ALL, (state,action) =>{
        console.log('Clearing all and reloading the page.');
        window.location.reload();
    });
    const unsubscribeEditJobDescription = store.subscribe(ACTION_TYPES.EDIT_JOB_DESCRIPTION, (state,action) =>{
        console.log('Edit job details.')
        localStorage.setItem('jobsData',JSON.stringify(state.jobs));
        populateJobTable(state.jobs,jobTable);
        //jobTable.row(state.selectedJobId).select();
    })

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
    const assignUncodableButton = document.getElementById('assign-uncodable');
    const clearButton = document.getElementById('clear-all');
    const downloadButton = document.getElementById('downloadButton');

    // Download event (doesn't affect state)
    downloadButton.addEventListener('click', () => {
        downloadCSV(store.getState().jobs,"results.csv");
    });

    // File chooser event
    fileInput.addEventListener('change', async (event) => {
        const newJobs = await handleFileSelect(event);
        store.dispatch(loadJobs(newJobs));
        fileInput.value = '';
    });
    // Backup in case file chooser 'change' event doesn't fire
    fileInput.addEventListener('click', () => {
        fileInput.value = '';
    });

    // Click search button event
    searchButton.addEventListener('click', async () => {
        resultsDataTable.clear().draw();
        resultsDataTable.row.add({label: 'Loading...', description: 'Please wait', distance: ''}).draw();
        const newResultsData = await fetchResults(store.getState().jobs);
        store.dispatch(updateResults(newResultsData));
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

    // Assign uncodable event
    assignUncodableButton.addEventListener('click', () => {
        const selectedJobId = store.getState().selectedJobId;
        const uncodableResult = {
            label: "*",
            description: "uncodable",
            distance: 9.99,
            rank: 9999,
        }
        if (selectedJobId) {
            store.dispatch(assignResult(selectedJobId, uncodableResult));
        }
    });

    // Clear All event
    clearButton.addEventListener('click', () =>{
        if (confirm('Are you sure you want to clear all data? This action cannot be undone.')) {
            store.dispatch(clearAll());
        }
    })
});
