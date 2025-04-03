// state.js

import { initialState } from './actions.js';

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
          hideCoded: JSON.parse(localStorage.getItem('hideCoded')) || initialState.hideCoded,
          sessionID: JSON.parse(localStorage.getItem('sessionID')) || null
      };
  } catch (e) {
      console.error('Error loading state:', e);
      return initialState;
  }
}

/**
 * Creates a store
 * @returns {Object} Store object with getState, dispatch, and subscribe methods
 */
 export function createStore() {
    let state = loadStateFromStorage();
    let listeners = new Map();

    const getState = () => state;

    const dispatch = (action) => {
        try {
          state = action.stateUpdater(state);
        } catch (error) {
          console.log("Error in state.js with undefined action.");
        }
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


export const store = createStore();
