import { notification } from 'antd';
import React, { useEffect, useLayoutEffect, useMemo, useState } from 'react';
import { DndProvider } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';
import { HelmetProvider } from 'react-helmet-async';

import Button from 'components/kit/Button';
import Link from 'components/Link';
import Navigation from 'components/Navigation';
import PageMessage from 'components/PageMessage';
import Router from 'components/Router';
import useAuthCheck from 'hooks/useAuthCheck';
import useKeyTracker from 'hooks/useKeyTracker';
import usePageVisibility from 'hooks/usePageVisibility';
import useResize from 'hooks/useResize';
import useRouteTracker from 'hooks/useRouteTracker';
import { SettingsProvider } from 'hooks/useSettingsProvider';
import useTelemetry from 'hooks/useTelemetry';
import useTheme from 'hooks/useTheme';
import Omnibar from 'omnibar/Omnibar';
import appRoutes from 'routes';
import { paths, serverAddress } from 'routes/utils';
import Spinner from 'shared/components/Spinner/Spinner';
import usePolling from 'shared/hooks/usePolling';
import { StoreContext } from 'stores';
import { useAuth } from 'stores/auth';
import { initInfo, useDeterminedInfo, useEnsureInfoFetched } from 'stores/determinedInfo';
import { useCurrentUser, useEnsureCurrentUserFetched, useFetchUsers } from 'stores/users';
import { correctViewportHeight, refreshPage } from 'utils/browser';
import { Loadable } from 'utils/loadable';

import css from './App.module.scss';

const AppView: React.FC = () => {
  const resize = useResize();
  const auth = useAuth().auth;
  const isAuthenticated = Loadable.match(auth, {
    Loaded: (auth) => auth.isAuthenticated,
    NotLoaded: () => false,
  });
  const loadableUser = useCurrentUser();
  const infoLoadable = useDeterminedInfo();
  const info = Loadable.getOrElse(initInfo, infoLoadable);
  const [canceler] = useState(new AbortController());
  const { updateTelemetry } = useTelemetry();
  const checkAuth = useAuthCheck();

  const isServerReachable = useMemo(() => {
    return Loadable.match(infoLoadable, {
      Loaded: (info) => !!info.clusterId,
      NotLoaded: () => undefined,
    });
  }, [infoLoadable]);

  const fetchInfo = useEnsureInfoFetched(canceler);
  const fetchUsers = useFetchUsers(canceler);
  const fetchCurrentUser = useEnsureCurrentUserFetched(canceler);

  useEffect(() => {
    if (isServerReachable) checkAuth();
  }, [checkAuth, isServerReachable]);

  useTheme();
  useKeyTracker();
  usePageVisibility();
  useRouteTracker();

  // Poll every 10 minutes
  usePolling(fetchInfo, { interval: 600000 });

  useEffect(() => {
    if (isAuthenticated) {
      fetchUsers();
      fetchCurrentUser();
    }
  }, [isAuthenticated, fetchCurrentUser, fetchUsers]);

  useEffect(() => {
    /*
     * Check to make sure the WebUI version matches the platform version.
     * Skip this check for development version.
     */

    if (!process.env.IS_DEV && info.version !== process.env.VERSION) {
      const btn = (
        <Button type="primary" onClick={refreshPage}>
          Update Now
        </Button>
      );
      const message = 'New WebUI Version';
      const description = (
        <div>
          WebUI version <b>v{info.version}</b> is available. Check out what&apos;s new in our&nbsp;
          <Link external path={paths.docs('/release-notes.html')}>
            release notes
          </Link>
          .
        </div>
      );
      notification.warn({
        btn,
        description,
        duration: 0,
        key: 'version-mismatch',
        message,
        placement: 'bottomRight',
      });
    }
  }, [info]);

  // Detect telemetry settings changes and update telemetry library.
  useEffect(() => {
    updateTelemetry(auth, loadableUser, info);
  }, [auth, loadableUser, info, updateTelemetry]);

  // Abort cancel signal when app unmounts.
  useEffect(() => {
    return () => canceler.abort();
  }, [canceler]);

  // Correct the viewport height size when window resize occurs.
  useLayoutEffect(() => correctViewportHeight(), [resize]);

  return Loadable.match(infoLoadable, {
    Loaded: () => (
      <div className={css.base}>
        {isServerReachable ? (
          <SettingsProvider>
            <Navigation>
              <main>
                <Router routes={appRoutes} />
              </main>
            </Navigation>
          </SettingsProvider>
        ) : (
          <PageMessage title="Server is Unreachable">
            <p>
              Unable to communicate with the server at &quot;{serverAddress()}&quot;. Please check
              the firewall and cluster settings.
            </p>
            <Button onClick={refreshPage}>Try Again</Button>
          </PageMessage>
        )}
        <Omnibar />
      </div>
    ),
    NotLoaded: () => <Spinner center />,
  });
};

const App: React.FC = () => {
  return (
    <HelmetProvider>
      <StoreContext>
        <DndProvider backend={HTML5Backend}>
          <AppView />
        </DndProvider>
      </StoreContext>
    </HelmetProvider>
  );
};

export default App;
