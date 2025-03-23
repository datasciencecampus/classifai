/**
 * @file uiService.js
 * @description Provides UI-related functions for the SIC/SOC Coding Tool.
 */
import { getResultsForJob } from './dataService.js';
import {selectJob, selectResult, assignResult, uncodableResult} from './actions.js'


/**
 * Initializes DataTables for jobs and code results.
 * @param {Object} store - A store object for state management
 * @returns {{jobTable: DataTables.Api, resultsDataTable: DataTables.Api}} The
 * initialized DataTables.
 */
export function initTables(store) {
    const jobTable = new DataTable(
        document.getElementById('job-table'),
        {
            select: 'single',
            columns: [
                { data: 'id', title: 'ID' },
                { data: 'description', title: 'Description' },
                { data: 'code', title: 'Code' }
            ],
            autoWidth: false,
            order: false,
            paging: true,
        }
    );

    const resultsDataTable = new DataTable(
        document.getElementById('results-table'),
        {
            select: 'single',
            columns: [
                { data: 'label', title: 'Code' },
                { data: 'description', title: 'Description' },
                { data: 'distance', title: 'Distance' },
            ],
            order: [[2, 'asc']],
            columnDefs: [{targets: [0,2], width: '20%'}],
            autoWidth: false,
            paging: true,
        }
    );

    // Make table tabbable
    jobTable.table().node().setAttribute('tabindex', '0');
    resultsDataTable.table().node().setAttribute('tabindex', '0');



    // Handle keyboard events
    jobTable.on('keydown', function(event) {
        if (event.key === 'Enter' || event.key === 'ArrowDown' || event.key === 'j' || event.key === 'J') {
            console.log(jobTable.row('.selected').index());
            incrementDataTable(jobTable, (rowData) => store.dispatch(selectJob(rowData.id)));
        } else if (event.key === 'ArrowUp' || event.key === 'k' || event.key === 'K') {
            incrementDataTable(jobTable, (rowData) => store.dispatch(selectJob(rowData.id)), -1);
        } else if (event.key === 'ArrowRight' || event.key === 'h' || event.key === 'H' || event.key === 'l' || event.key === 'L') {
            resultsDataTable.row(0).select();
            resultsDataTable.table().node().focus();
        } else if (event.key === 'y' || event.key === 'Y') {
            const state = store.getState();
            if (state.selectedJobId && state.selectedResult.label) {
                const jobID = state.selectedJobId;
                const result = state.selectedResult;
                incrementDataTable(jobTable, (rowData) => store.dispatch(selectJob(rowData.id)));
                store.dispatch(assignResult(jobID, result));
           }
        }  else if (event.key === 'u' || event.key === 'U') {
            const selectedJobId = store.getState().selectedJobId;
            if (selectedJobId) {
                incrementDataTable(jobTable, (rowData) => store.dispatch(selectJob(rowData.id)));
                store.dispatch(assignResult(selectedJobId, uncodableResult));
            }
        }else if (event.key === 'n' || event.key === 'N') {
            const clearData = {
                label: "",
                description: "",
                distance: "",
                rank: "",
            }
            store.dispatch(assignResult(store.getState().selectedJobId,clearData));
        }
    });

    resultsDataTable.on('keydown', function(event) {
        if (event.key === 'Enter' || event.key === 'ArrowDown' || event.key === 'j' || event.key === 'J') {
            incrementDataTable(resultsDataTable,(rowData) => store.dispatch(selectResult(rowData)));
        } else if (event.key === 'ArrowUp' || event.key === 'k' || event.key === 'K') {
            incrementDataTable(resultsDataTable,(rowData) => store.dispatch(selectResult(rowData)), -1);
        } else if (event.key === 'ArrowLeft' || event.key === 'h' || event.key === 'H' || event.key === 'l' || event.key === 'L') {
            jobTable.row('.selected').select();
            jobTable.table().node().focus();
        }  else if (event.key === 'y' || event.key === 'Y') {
            const state = store.getState();
            if (state.selectedJobId && state.selectedResult.label) {
                const jobID = state.selectedJobId;
                const result = state.selectedResult;
                incrementDataTable(jobTable, (rowData) => store.dispatch(selectJob(rowData.id)));
                store.dispatch(assignResult(jobID, result));
            }
        }  else if (event.key === 'u' || event.key === 'U') {
            const selectedJobId = store.getState().selectedJobId;
            if (selectedJobId) {
                incrementDataTable(jobTable, (rowData) => store.dispatch(selectJob(rowData.id)));
                store.dispatch(assignResult(selectedJobId, uncodableResult));
            }
        }
    });
    // Handle click events
    jobTable.on('click', 'tbody tr', function() {
        // jobTable.row(this).select();
        // jobTable.table().node().focus();
        const rowData = jobTable.row(this).data();
        store.dispatch(selectJob(rowData.id));
    });
    resultsDataTable.on('click', 'tbody tr', function() {
        const rowData = resultsDataTable.row(this).data() || {};
        store.dispatch(selectResult(rowData));
    });

    return { jobTable, resultsDataTable };
}

/**
 * Updates the job table with new data.
 * @param {Array} jobsData - The array of job objects.
 * @param {DataTables.Api} jobTable - The job DataTable instance.
 *
 * Checks for a flag in 'hideCodedRows', creating a new array with the coded
 * rows removed if true.
 */
export function populateJobTable(jobsData, jobTable, hideCodedRows) {
    jobTable.clear();
    if (hideCodedRows == true) {
        let newJobsData = []
        for (let job of jobsData) {
            if (String(job.code).length > 0) {
                continue;
            }
            else {
                newJobsData.push(job);
            };
        };
        jobsData = newJobsData;
    };
    jobTable.rows.add(jobsData.map(job => ({
        id: job.id,
        description: job.description,
        code: job.code
    }))).draw(false);
}

/**
 * Displays job details in the UI above the results table.
 * @param {Object} job - The job object to display.
 * @param {Function} onSave - Callback function when saving edited description
 */
export function showJobDetails(job, onSave) {
    const jobDetailsContent = document.getElementById('job-details');

    // Flag to track if we're in edit mode
    let isEditing = false;

    function createViewMode(job) {
        return `
                <h2>${job.description || 'Select a row...'}</h2>
                ${job.description ? `
                    <button class="edit-btn" id="edit-button">Edit</button>
                ` : ''}
        `;
    }

    function createEditMode(job) {
        return `
                <input
                    type="text"
                    id="edit-description"
                    value="${job.description || ''}"
                    class="edit-input"
                >
                <div class="edit-buttons">
                    <button class="ok-btn" id="ok-button">Ok</button>
                    <button class="cancel-btn" id="cancel-button">Cancel</button>
                </div>
        `;
    }

    function render(job) {
        jobDetailsContent.innerHTML = isEditing ? createEditMode(job) : createViewMode(job);

        if (isEditing) {
            // Add event listeners for edit mode
            const okButton = document.getElementById('ok-button');
            const cancelButton = document.getElementById('cancel-button');
            const inputField = document.getElementById('edit-description');

            okButton.addEventListener('click', () => {
                const newDescription = inputField.value.trim();
                if (newDescription !== job.description) {
                    job = { ...job, description: newDescription};
                    // Call the provided callback with updated value
                    onSave(job);
                }
                isEditing = false;
                render(job);
            });

            cancelButton.addEventListener('click', () => {
                isEditing = false;
                render(job);
            });
        } else {
            // Add event listener for view mode
            const editButton = document.getElementById('edit-button');
            if (editButton) {
                editButton.addEventListener('click', () => {
                    isEditing = true;
                    render(job);
                });
            }
        }
    }

    // Initial render
    render(job);
}


/**
 * Updates the code results table for the currently selected job.
 * @param {number} selectedJobId - The ID of the job to show results for.
 * @param {Array} resultsData - the full results data.
 * @param {DataTables.Api} resultsDataTable - the results DataTable instance
 */
export function updateResultsTable(selectedJobId,resultsData,resultsDataTable) {

    try {
        resultsDataTable.clear();
        if (selectedJobId && resultsData) {
            const results = getResultsForJob(selectedJobId, resultsData);
            if (results && results.length > 0) {
                resultsDataTable.rows.add(results);
            }
        }
        resultsDataTable.draw(false);

        // Select and click the first row
        if (resultsDataTable.rows().count() > 0) {
            resultsDataTable.row(0).select();

            // Get the DOM node of the first row and trigger a click event
            const firstRow = resultsDataTable.row(0).node();
            if (firstRow) {
                firstRow.dispatchEvent(new Event('click', { bubbles: true }));
            }
        }
    } catch (error) {
        console.error('Error in updateResultsTable:', error);
        resultsDataTable.clear().draw();
    }
}
/**
* Procedure which increments the selection on the jobTable datatable
*
* @param {DataTables.Api} dataTable - The datatable instance
* @param {function} updateCallback - The state management callback. Function of 'rowData'
* @param {int} nRowsDown - the number of rows to move down (negative to move up)
*/
export function incrementDataTable(dataTable, updateCallback,nRowsDown=1) {
        const currentRowIndex = dataTable.row('.selected').index();
        const visibleRows = dataTable.rows({order: 'current'}).indexes();
        const currentPosition = visibleRows.indexOf(currentRowIndex);
        const nextPosition = currentPosition + nRowsDown;
        if (nextPosition >= 0 && nextPosition < visibleRows.length) {
            dataTable.row(visibleRows[nextPosition]).select();
            const rowData = dataTable.row(visibleRows[nextPosition]).data();
            updateCallback(rowData);
         }
}
