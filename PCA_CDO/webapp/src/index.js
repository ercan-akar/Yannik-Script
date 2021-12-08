import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import * as serviceWorker from './serviceWorker';
import Plant from './Plant';
import {
  BrowserRouter as Router,
  Route,
  Switch,
  Link,
  useParams
} from "react-router-dom";

ReactDOM.render(
  <Router>
    <Plant />
  </Router>,
  document.getElementById('root')
);

// If you want your app to work offline and load faster, you can change
// unregister() to register() below. Note this comes with some pitfalls.
// Learn more about service workers: https://bit.ly/CRA-PWA
serviceWorker.unregister();