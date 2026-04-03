import React, {useState, useEffect, useRef} from 'react';
import { useNavigate } from 'react-router-dom';
import "./pages-styles/Call.css";
import {useReactMediaRecorder } from 'react-media-recorder';

export default function Call(){
    const [prediction, setPrediction] = useState("");
    const [confidence, setConfidence] = useState(0);
    const [recording, setRecording] = useState(false);
    const [audioContext, setAudioContext] = useState(null);
    const [processor, setProcessor] = useState(null);
    const [source, setSource] = useState(null);
    const [status, setStatus] = useState("");
    const audioBufferRef = useRef([]);
    const recordingRef = useRef(false);
    const audioContextRef = useRef(null);
    const processorRef = useRef(null);
    const sourceRef = useRef(null);
    const streamRef = useRef(null);
    const [recorder, setRecorder] = useState(null);
    const [stream, setStream] = useState(null);
    let lastSendTime = 0;
    const navigate = useNavigate();

    const sending = useRef(false);
    const queue = useRef([]);

    const sendChunk = async (blob) => {
       
        queue.current.push(blob);
        if (sending.current) return;
        sending.current = true;

        while (queue.current.length > 0) {
            const current = queue.current.shift();
            const formData = new FormData();
            formData.append("chunk", current);

            try {
                const res = await fetch("http://127.0.0.1:8000/predict", {
                    method: "POST",
                    body: formData,
                    mode: "cors",
                });
                if (!res.ok) throw new Error(`Status ${res.status}`);
                const data = await res.json();
                setPrediction(data.prediction);
                setConfidence(data.confidence);
                setStatus(data.status);
                if (data.status === "SOS DETECTED SENDING HELP") {
                    console.log("🚨 SOS TRIGGERED");
                }
            } catch (err) {
                console.error("Chunk send failure:", err);
            }
        }

        sending.current = false;
    };
    const sendPCM = async (pcmArray) => {
        try {
            const res = await fetch("http://127.0.0.1:8000/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/octet-stream"
                },
                body: pcmArray.buffer
            });

            const data = await res.json();

            setPrediction(data.prediction);
            setConfidence(data.confidence);
            setStatus(data.status);

            if (data.status === "SOS DETECTED SENDING HELP") {
                console.log("🚨 SOS TRIGGERED");
                alert("SOS Triggered");
            }

        } catch (err) {
            console.error("PCM send error:", err);
        }
    };

    const handleStart = async () => {
        if (recordingRef.current) {
            console.log("Already recording");
            return;
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

            const audioContext = new AudioContext({ sampleRate: 16000 });
            const source = audioContext.createMediaStreamSource(stream);
            const processor = audioContext.createScriptProcessor(4096, 1, 1);

            source.connect(processor);
            processor.connect(audioContext.destination);

            audioBufferRef.current = [];

            processor.onaudioprocess = (e) => {
                if (!recordingRef.current) return;

                const input = e.inputBuffer.getChannelData(0);

                const pcm = new Int16Array(input.length);
                for (let i = 0; i < input.length; i++) {
                    pcm[i] = Math.max(-1, Math.min(1, input[i])) * 32767;
                }

                audioBufferRef.current.push(...pcm);

                if (audioBufferRef.current.length >= 16000) {
                    const chunk = new Int16Array(
                        audioBufferRef.current.slice(0, 16000)
                    );

                    sendPCM(chunk);

                    // overlap (important for accuracy)
                    audioBufferRef.current = audioBufferRef.current.slice(8000);
                }
            };

            // save refs
            streamRef.current = stream;
            audioContextRef.current = audioContext;
            processorRef.current = processor;
            sourceRef.current = source;

            recordingRef.current = true;
            setRecording(true);

            console.log("🎤 Recording started");

        } catch (err) {
            console.error("Mic error:", err);
        }
    };
    const handleStop = () => {
        try {
            recordingRef.current = false;

            if (processorRef.current) {
                processorRef.current.onaudioprocess = null;
                processorRef.current.disconnect();
            }

            if (sourceRef.current) {
                sourceRef.current.disconnect();
            }

            if (audioContextRef.current) {
                audioContextRef.current.close();
            }

            if (streamRef.current) {
                streamRef.current.getTracks().forEach(track => track.stop());
            }

            audioBufferRef.current = [];

            setRecording(false);
            console.log("🛑 FULL STOP");

        } catch (err) {
            console.error("Stop error:", err);
        }
    };

    const stars = Array.from({length:175}).map((_,index)=>{
        const top = Math.random()*100;
        const left = Math.random()*100;
        const delay = Math.random()*5+1;
        const size = Math.random()*3;
        return(
            <span key = {index} className = 'star' style = {{
                top : `${top}%`,
                left : `${left}%`,
                animationDelay : `${delay}s`,
                width : `${size}px`,
                height : `${size}px`,
            }}/>
        );
    });
    return (
        <div>
            <div className="callContainer">
                {stars}
                <div className ="caller">572-299-2934</div>
                <div className = "caller-name">Krish</div>
                <button className="end-call" onClick={()=>{
                    navigate('/');
                    handleStop();
                }}>.</button>
                <button className="mute-call">.</button>
                <button className="speaker" onClick={handleStart}>.</button>
                <div className = "prediction">Prediction: {prediction}</div>
                <div className = "confidence">Confidence: {Number(confidence).toFixed(2)}</div>
                <div className="status">{status}</div>
            </div>
        </div>
    )
}