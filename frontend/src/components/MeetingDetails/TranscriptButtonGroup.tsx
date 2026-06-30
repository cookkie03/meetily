"use client";

import { useState, useCallback, useEffect, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { ButtonGroup } from '@/components/ui/button-group';
import { Copy, FolderOpen, Loader2, RefreshCw } from 'lucide-react';
import Analytics from '@/lib/analytics';
import { RetranscribeDialog } from './RetranscribeDialog';
import { useConfig } from '@/contexts/ConfigContext';
import { invoke } from '@tauri-apps/api/core';
import { listen, UnlistenFn } from '@tauri-apps/api/event';
import { toast } from 'sonner';


interface TranscriptButtonGroupProps {
  transcriptCount: number;
  onCopyTranscript: () => void;
  onOpenMeetingFolder: () => Promise<void>;
  meetingId?: string;
  meetingFolderPath?: string | null;
  onRefetchTranscripts?: () => Promise<void>;
}


export function TranscriptButtonGroup({
  transcriptCount,
  onCopyTranscript,
  onOpenMeetingFolder,
  meetingId,
  meetingFolderPath,
  onRefetchTranscripts,
}: TranscriptButtonGroupProps) {
  const { betaFeatures } = useConfig();
  const [showRetranscribeDialog, setShowRetranscribeDialog] = useState(false);
  const [isIdentifyingSpeakers, setIsIdentifyingSpeakers] = useState(false);
  const [diarizationMessage, setDiarizationMessage] = useState<string | null>(null);
  const [downloadProgress, setDownloadProgress] = useState<number | null>(null);
  const isIdentifyingRef = useRef(false);

  useEffect(() => {
    isIdentifyingRef.current = isIdentifyingSpeakers;
  }, [isIdentifyingSpeakers]);

  useEffect(() => {
    let unlistenProgress: UnlistenFn | undefined;
    let unlistenComplete: UnlistenFn | undefined;
    let unlistenDownload: UnlistenFn | undefined;
    let cancelled = false;

    const setupListeners = async () => {
      try {
        unlistenProgress = await listen<{
          meeting_id?: string;
          status?: string;
          progress?: number;
          message?: string;
        }>('diarization-progress', async (event) => {
          const payload = event.payload;
          if (!meetingId || payload.meeting_id !== meetingId) return;

          if (payload.message) {
            setDiarizationMessage(payload.message);
          }
          if (typeof payload.progress === 'number') {
            setDownloadProgress(null);
          }
          if (payload.status === 'failed') {
            setIsIdentifyingSpeakers(false);
            isIdentifyingRef.current = false;
            setDiarizationMessage(null);
            toast.error(payload.message ?? 'Speaker identification failed');
          } else {
            setIsIdentifyingSpeakers(true);
            isIdentifyingRef.current = true;
          }
        });
        if (cancelled) {
          unlistenProgress();
          return;
        }

        unlistenComplete = await listen<{ meeting_id?: string }>('diarization-complete', async (event) => {
          const payload = event.payload;
          if (!meetingId || payload.meeting_id !== meetingId) return;

          setIsIdentifyingSpeakers(false);
          isIdentifyingRef.current = false;
          setDownloadProgress(null);

          try {
            if (onRefetchTranscripts) {
              await onRefetchTranscripts();
            }
            toast.success('Speaker identification complete');
            setDiarizationMessage(null);
          } catch (error) {
            console.error('Failed to refresh transcripts after diarization:', error);
            toast.error('Speaker identification completed, but refresh failed');
            setDiarizationMessage(null);
          }
        });
        if (cancelled) {
          unlistenComplete();
          return;
        }

        unlistenDownload = await listen<{
          model?: string;
          progress?: number;
          downloaded_bytes?: number;
          total_bytes?: number;
        }>('diarization-model-download-progress', (event) => {
          if (!isIdentifyingRef.current) return;

          const payload = event.payload;
          if (typeof payload.progress === 'number') {
            setDownloadProgress(payload.progress);
            setDiarizationMessage(
              payload.model
                ? `Downloading ${payload.model} model... ${payload.progress}%`
                : `Downloading diarization model... ${payload.progress}%`
            );
          }
        });
        if (cancelled) {
          unlistenDownload();
        }
      } catch (error) {
        if (!cancelled) {
          console.error('Failed to set up diarization listeners:', error);
        }
      }
    };

    setupListeners();

    return () => {
      cancelled = true;
      unlistenProgress?.();
      unlistenComplete?.();
      unlistenDownload?.();
    };
  }, [meetingId, onRefetchTranscripts]);

  const handleRetranscribeComplete = useCallback(async () => {
    // Refetch transcripts to show the updated data
    if (onRefetchTranscripts) {
      await onRefetchTranscripts();
    }
  }, [onRefetchTranscripts]);

  const handleIdentifySpeakers = useCallback(async () => {
    if (!meetingId || isIdentifyingSpeakers) return;

    setIsIdentifyingSpeakers(true);
    isIdentifyingRef.current = true;
    setDiarizationMessage('Starting speaker identification...');
    setDownloadProgress(null);

    try {
      await invoke('run_diarization', { meeting_id: meetingId });
    } catch (error) {
      console.error('Failed to start diarization:', error);
      setIsIdentifyingSpeakers(false);
      isIdentifyingRef.current = false;
      setDiarizationMessage(null);
      toast.error('Failed to start speaker identification');
    }
  }, [isIdentifyingSpeakers, meetingId]);

  return (
    <div className="flex flex-col items-center justify-center w-full gap-2">
      <ButtonGroup>
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            Analytics.trackButtonClick('copy_transcript', 'meeting_details');
            onCopyTranscript();
          }}
          disabled={transcriptCount === 0}
          title={transcriptCount === 0 ? 'No transcript available' : 'Copy Transcript'}
        >
          <Copy />
          <span className="hidden lg:inline">Copy</span>
        </Button>

        <Button
          size="sm"
          variant="outline"
          className="xl:px-4"
          onClick={() => {
            Analytics.trackButtonClick('open_recording_folder', 'meeting_details');
            onOpenMeetingFolder();
          }}
          title="Open Recording Folder"
        >
          <FolderOpen className="xl:mr-2" size={18} />
          <span className="hidden lg:inline">Recording</span>
        </Button>

        {betaFeatures.importAndRetranscribe && meetingId && meetingFolderPath && (
          <Button
            size="sm"
            variant="outline"
            className="bg-gradient-to-r from-blue-50 to-purple-50 hover:from-blue-100 hover:to-purple-100 border-blue-200 xl:px-4"
            onClick={() => {
              Analytics.trackButtonClick('enhance_transcript', 'meeting_details');
              setShowRetranscribeDialog(true);
            }}
            title="Retranscribe to enhance your recorded audio"
          >
            <RefreshCw className="xl:mr-2" size={18} />
            <span className="hidden lg:inline">Enhance</span>
          </Button>
        )}

        {meetingId && (
          <Button
            size="sm"
            variant="outline"
            className="xl:px-4"
            onClick={() => {
              Analytics.trackButtonClick('identify_speakers', 'meeting_details');
              void handleIdentifySpeakers();
            }}
            disabled={isIdentifyingSpeakers}
            title="Identify and label speakers in the transcript"
          >
            <Loader2 className={`xl:mr-2 ${isIdentifyingSpeakers ? 'animate-spin' : ''}`} size={18} />
            <span>Identify speakers</span>
          </Button>
        )}
      </ButtonGroup>

      {(isIdentifyingSpeakers || diarizationMessage) && (
        <div className="flex items-center gap-2 text-xs text-gray-500">
          <div className="h-3 w-3 rounded-full bg-blue-500 animate-pulse" />
          <span className="max-w-[22rem] truncate">
            {diarizationMessage ?? 'Identifying speakers...'}
            {downloadProgress !== null ? ` (${downloadProgress}%)` : ''}
          </span>
        </div>
      )}

      {betaFeatures.importAndRetranscribe && meetingId && meetingFolderPath && (
        <RetranscribeDialog
          open={showRetranscribeDialog}
          onOpenChange={setShowRetranscribeDialog}
          meetingId={meetingId}
          meetingFolderPath={meetingFolderPath}
          onComplete={handleRetranscribeComplete}
        />
      )}
    </div>
  );
}
