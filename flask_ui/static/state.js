// state.js

import { ACTION_TYPES } from './actions.js';

/**
 * The initial state of the application
 */
const initialState = {
  jobs: [],
  selectedJobId: null,
  resultsData: [],
  selectedResult: {},
};


/**
 * Utility function for upsert in Array
 * @param {Array} data - Data Array with each record having an 'id' (or other comparison field - see `matchFunction`)
 * @param {Object} newRecord - New record to update or insert
 * @param {function} matchFunction - Function determining fields to compare on
 * @returns {Array} updated data
 */
const upsertRecord = (data, newRecord, matchFunction = (a,b) => a.id === b.id) => {
  let found = false;
  const updated = data.map(record => {
      if (matchFunction(record, newRecord)) {
          found = true;
          return newRecord;
      }
      return record;
  });
  return found ? updated : [...updated, newRecord];
};

/**
 * Reducer function to handle state changes based on actions
 * @param {Object} state - Current state
 * @param {Object} action - Action object
 * @returns {Object} New state
 */
export function reducer(state, action) {
    switch (action.type) {
      case ACTION_TYPES.LOAD_JOBS:
        return { ...state, jobs: action.payload };
      case ACTION_TYPES.SELECT_JOB:
        return { ...state, selectedJobId: action.payload };
      case ACTION_TYPES.UPDATE_RESULTS:
        return { ...state, resultsData: action.payload };
      case ACTION_TYPES.SELECT_RESULT:
        return { ...state, selectedResult: action.payload };
      case ACTION_TYPES.CLEAR_ALL:
        return initialState;
      case ACTION_TYPES.UPDATE_ONE_RESULT:
        return {
          ...state,
          resultsData: upsertRecord(
            state.resultsData,
            action.payload,
            (result, newresult) => result.input_id === newresult.input_id,
          )
        };
      case ACTION_TYPES.EDIT_JOB_DESCRIPTION:
        return {
            ...state,
            jobs: upsertRecord(
              state.jobs,
              action.payload,
            )
            };
      case ACTION_TYPES.ASSIGN_RESULT:
        return {
            ...state,
            jobs: state.jobs.map(job =>
                job.id === action.payload.jobId
                    ? { ...job,
                        sic_code: action.payload.result.label,
                        sic_code_description: action.payload.result.description,
                        sic_code_score: action.payload.result.distance,
                        sic_code_rank: action.payload.result.rank,
                    }
                    : job
            )
        };
      default:
        return state;
    }
}

/**
 * @typedef {Object} AppState
 * @property {Array} jobs - Array of job objects
 * @property {Array} resultsData - Array of results data
 * @property {string|null} selectedJobId - ID of selected job or null
 * @property {Object} selectedResult - Currently selected result object
 */

/**
 * Loads application state from localStorage.
 *
 * Returns initial state object with empty values if localStorage is empty
 * or if there's an error parsing the stored data.
 *
 * @returns {AppState} The application state loaded from localStorage
 */
const loadStateFromStorage = () => {
  try {
      return {
          jobs: JSON.parse(localStorage.getItem('jobsData')) || [],
          resultsData: JSON.parse(localStorage.getItem('resultsData')) || [],
          selectedJobId: JSON.parse(localStorage.getItem('selectedJobId')) || null,
          selectedResult: JSON.parse(localStorage.getItem('selectedResult')) || {},
      };
  } catch (e) {
      console.error('Error loading state:', e);
      return initialState;
  }
}

/**
 * Creates a store with the given reducer function
 * @param {function} reducer - Function that returns new state based on current state and action
 * @returns {Object} Store object with getState, dispatch, and subscribe methods
 */
 export function createStore(reducer) {
    let state = loadStateFromStorage();
    let listeners = new Map();

    const getState = () => state;

    const dispatch = (action) => {
        state = reducer(state, action);
        if (listeners.has(action.type)) {
            listeners.get(action.type).forEach(listener => listener(state, action));
        }
        if (listeners.has('*')) {
            listeners.get('*').forEach(listener => listener(state, action));
        }
    };

    const subscribe = (actionType, listener) => {
        if (!listeners.has(actionType)) {
            listeners.set(actionType, []);
        }
        listeners.get(actionType).push(listener);
        return () => {
            if (listeners.has(actionType)) {
                listeners.set(actionType, listeners.get(actionType).filter(l => l !== listener));
                if (listeners.get(actionType).length === 0) {
                    listeners.delete(actionType);
                }
            }
        };
    };

    return { getState, dispatch, subscribe };
}


export const store = createStore(reducer);
