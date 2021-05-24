import React, { useEffect, useState } from 'react';
import logo from './logo.svg';
import './App.css';
import * as tf from '@tensorflow/tfjs';
import padSequences from './helper/paddedSeq'
import Emoji from 'a11y-react-emoji'
import {
  TextField,
  Grid,
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Button
} from '@material-ui/core'
import MenuIcon from '@material-ui/icons/Menu';
import { makeStyles } from '@material-ui/core/styles';
import { blue } from '@material-ui/core/colors';



function App() {
  const useStyles = makeStyles((theme) => ({
    root: {
      flexGrow: 1,
    },
    menuButton: {
      marginRight: theme.spacing(2),
    },
    title: {
      flexGrow: 1,
    },
  }));
  const classes = useStyles();

  const url = {

    model: 'https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/model.json',
    metadata: 'https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/metadata.json'
};

const OOV_INDEX = 2;

const [metadata, setMetadata] = useState();
const [model, setModel] = useState();
const [testText, setText] = useState("");
const [testScore, setScore] = useState("");
const [trimedText, setTrim] = useState("")
const [seqText, setSeq] = useState("")
const [padText, setPad] = useState("")
const [inputText, setInput] = useState("")


async function loadModel(url) {
  try {
    const model = await tf.loadLayersModel(url.model);
    setModel(model);
  } catch (err) {
    console.log(err);
  }
}

async function loadMetadata(url) {
  try {
    const metadataJson = await fetch(url.metadata);
    const metadata = await metadataJson.json();
    setMetadata(metadata);
  } catch (err) {
    console.log(err);
  }
}


const getSentimentScore =(text) => {
  console.log(text)
  const inputText = text.trim().toLowerCase().replace(/(\.|\,|\!)/g, '').split(' ');
  setTrim(inputText)
  console.log(inputText)
  const sequence = inputText.map(word => {
    let wordIndex = metadata.word_index[word] + metadata.index_from;
    if (wordIndex > metadata.vocabulary_size) {
      wordIndex = OOV_INDEX;
    }
    return wordIndex;
  });
  setSeq(sequence)
  console.log(sequence)
  // Perform truncation and padding.
  const paddedSequence = padSequences([sequence], metadata.max_len);
  console.log(metadata.max_len)
  setPad(paddedSequence)

  const input = tf.tensor2d(paddedSequence, [1, metadata.max_len]);
  console.log(input)
  setInput(input)
  const predictOut = model.predict(input);
  const score = predictOut.dataSync()[0];
  predictOut.dispose();
  setScore(score)  
  return score;
}

function whenTyping(e) {
  setText(e.target.value);
  getSentimentScore(testText)
}

useEffect(()=>{
  tf.ready().then(
    ()=>{
      loadModel(url)
      loadMetadata(url)
    }
  );

},[])

  return (
     <>
       <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" className={classes.title}>
            Live Sentiment Analyzer
          </Typography>
        </Toolbar>
      </AppBar>
      <Grid container style={{ height:"90vh", padding:250}}>
      <Grid item lg={6} xs={12} style={{display:"flex",alignItems:"center", justifyContent:"center"}}>
      <TextField
          id="standard-read-only-input"
          label="Type a sentence here"
          onChange={(e)=>whenTyping(e)}
          defaultValue=""
          fullWidth
          value={testText}
          variant="outlined"
        />
        <br/>
        <br/>
      </Grid>
<Grid item lg={6} xs ={12} style={{display:"flex", alignItems:"center", justifyContent:"center"}}>
  <br/>
{testScore > 0.6?<><Typography style={{ fontSize: '45px' }}>
  <Emoji symbol="😀" label="Happy"/>
  </Typography></>:<></>}
{testScore < 0.4 && testScore !== ""?<><Typography style={{ fontSize: '45px' }}> 
  <Emoji symbol="🙁" label="Sad"/>
  </Typography></>:<></>}
{(testScore > 0.4 && testScore < 0.6) || testScore == ""?<><Typography style={{ fontSize: '45px' }}> 
  <Emoji symbol="😐" label="Neutral"/>
  </Typography></>:<></>}

</Grid>
      </Grid>

     </>
    
  );
}

export default App;